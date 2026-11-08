// ME193 Finger Lock (AirFour): button, servo latch, LED matrix and onboard RGB LEDs.
// The Mac reads the fingers and runs the lock policy; this sketch only shows
// what it is told and reports the button. Everything is millis()-based: there
// is no delay() in the loop, so a button press or a command is never missed.
//
// Wiring:
//   Micro servo orange (signal) -> D9, red -> 5V, brown -> GND
//   Arcade button microswitch   -> NO to D2, COM to GND (internal pull-up, pressed = LOW)
//   Onboard RGB LEDs LED3 and LED4 (active low: LOW = on)
//
// Links (via the Python side, python/main.py, which talks MQTT to the Mac):
//   Bridge.notify("button", ms)          arcade button pressed
//   Bridge.notify("lock_state", name)    locked / reading / unlocked / denied / lockout
//   Bridge.notify("lock_status", text)   log lines (also printed to Monitor as '#' lines)
//   Bridge.provide("lock_cmd", cmd, a, b)  commands from the Mac (see python/lock_protocol.py):
//       armed | digit a=slot b=value | unlock a=seconds | deny | timeout | lockout a=seconds | locked
//       servo a=angle (tuning only: lock/servo_test.py)
//
// Matrix (8 x 13):
//   locked    scrolls "LOCKED"          unlocked  scrolls "UNLOCKED"
//   button    all on briefly, then 3 empty slots; each accepted digit fills a slot with a dot
//   wrong     an X flashes 3 times       lockout   seconds left, LED4 blinking red fast
//
// Libraries: Arduino_RouterBridge, Servo >= 1.3.0, Arduino_LED_Matrix (with the board package).

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Servo.h>

// ---------- tune these ----------
const int LOCKED_ANGLE   = 0;     // servo angle with the latch closed (measured on the mounted latch)
const int UNLOCKED_ANGLE = 90;    // servo angle with the latch open (measured on the mounted latch)
const bool DEBUG = false;         // true: slots show the real digits (the Mac must also send them:
                                  // "send_digits": true in config/lock.json)
const bool MATRIX_FLIP = true;    // true: everything on the matrix is rotated 180 degrees
                                  // (the board is mounted upside down in the box)

// ---------- pins ----------
const int SERVO_PIN  = 9;
const int BUTTON_PIN = 2;

// ---------- timing (ms) ----------
const unsigned long DEBOUNCE_MS = 30;
const unsigned long SCROLL_STEP_MS = 90;      // one column per step
const unsigned long FLASH_MS = 250;           // whole matrix on after the button
const unsigned long X_ON_MS = 300, X_OFF_MS = 200;
const int X_FLASHES = 3;
const unsigned long BLINK_MS = 100;           // lockout red blink half-period
const unsigned long DEFAULT_RELOCK_MS = 10000;
const unsigned long ARMED_SAFETY_MS = 15000;  // reading with no word from the Mac: back to locked
const unsigned long HEARTBEAT_MS = 30000;

// Types used by function signatures: declared before any function, because the
// Arduino builder puts its generated prototypes above the first function.
enum Mode { M_LOCKED, M_FLASH, M_READING, M_UNLOCKED, M_DENIED, M_LOCKOUT };
struct Cmd { char name[12]; int a, b; };

Servo latch;
Arduino_LED_Matrix matrix;

// ---------- 5x7 font: 5 columns per glyph, bit 0 = top row ----------
struct Glyph { char c; uint8_t cols[5]; };
const Glyph FONT[] = {
  {' ', {0x00, 0x00, 0x00, 0x00, 0x00}},
  {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}}, {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
  {'2', {0x42, 0x61, 0x51, 0x49, 0x46}}, {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}},
  {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}}, {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
  {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}}, {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
  {'8', {0x36, 0x49, 0x49, 0x49, 0x36}}, {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
  {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}}, {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}},
  {'E', {0x7F, 0x49, 0x49, 0x49, 0x41}}, {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}},
  {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}}, {'N', {0x7F, 0x04, 0x08, 0x10, 0x7F}},
  {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}}, {'U', {0x3F, 0x40, 0x40, 0x40, 0x3F}},
};
// 3x5 digits for DEBUG slots: 3 columns, bit 0 = top row
const uint8_t SMALL_DIGITS[6][3] = {
  {0x1F, 0x11, 0x1F}, {0x12, 0x1F, 0x10}, {0x1D, 0x15, 0x17},
  {0x15, 0x15, 0x1F}, {0x07, 0x04, 0x1F}, {0x17, 0x15, 0x1D},
};

const uint8_t *glyph(char c) {
  for (const Glyph &g : FONT) if (g.c == c) return g.cols;
  return FONT[0].cols;
}

// ---------- frame buffer ----------
uint8_t frame[8][13];
void clearFrame() { memset(frame, 0, sizeof(frame)); }
// Every picture goes through here, so MATRIX_FLIP rotates all of them (and scrolling,
// rotated 180 degrees, still reads left to right for someone facing the flipped board).
void show() {
  if (!MATRIX_FLIP) {
    matrix.renderBitmap(frame, 8, 13);
    return;
  }
  static uint8_t rotated[8][13];
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 13; c++) rotated[r][c] = frame[7 - r][12 - c];
  matrix.renderBitmap(rotated, 8, 13);
}
void drawColumn(int col, uint8_t bits, int topRow, int height) {
  if (col < 0 || col > 12) return;
  for (int r = 0; r < height; r++) if (bits & (1 << r)) frame[topRow + r][col] = 1;
}

// ---------- scrolling text (non-blocking) ----------
uint8_t scrollCols[96];
int scrollLen = 0, scrollPos = 0;
unsigned long lastScrollMs = 0;

void setScrollText(const char *text) {
  scrollLen = 0;
  for (const char *p = text; *p && scrollLen < 80; p++) {
    const uint8_t *g = glyph(*p);
    for (int i = 0; i < 5; i++) scrollCols[scrollLen++] = g[i];
    scrollCols[scrollLen++] = 0;
  }
  for (int i = 0; i < 7; i++) scrollCols[scrollLen++] = 0;  // gap before it repeats
  scrollPos = 0;
}

void drawScroll() {
  clearFrame();
  for (int c = 0; c < 13; c++) drawColumn(c, scrollCols[(scrollPos + c) % scrollLen], 0, 7);
  show();
}

// ---------- other pictures ----------
int slotValue[3];      // 0 = empty, -1 = filled (dot), 1-5 = digit (DEBUG)

void drawSlots() {
  clearFrame();
  // Like a password field: each slot is an underline (columns 0-2, 5-7, 10-12, row 7);
  // an accepted digit puts a 3x3 dot above it (rows 3-5), or with DEBUG the digit (rows 1-5).
  for (int s = 0; s < 3; s++) {
    int left = s * 5;
    for (int i = 0; i < 3; i++) frame[7][left + i] = 1;
    if (slotValue[s] > 0 && DEBUG) {
      for (int i = 0; i < 3; i++) drawColumn(left + i, SMALL_DIGITS[slotValue[s]][i], 1, 5);
    } else if (slotValue[s] != 0) {
      for (int r = 3; r <= 5; r++) for (int i = 0; i < 3; i++) frame[r][left + i] = 1;
    }
  }
  show();
}

void drawAll(bool on) {
  memset(frame, on ? 1 : 0, sizeof(frame));
  show();
}

void drawX() {
  clearFrame();
  for (int i = 0; i < 8; i++) { frame[i][2 + i] = 1; frame[i][9 - i] = 1; }
  show();
}

void drawNumber(int n) {
  clearFrame();
  char text[4];
  snprintf(text, sizeof(text), "%d", n);
  int width = strlen(text) * 6 - 1;
  int left = (13 - width) / 2;
  for (int k = 0; text[k]; k++) {
    const uint8_t *g = glyph(text[k]);
    for (int i = 0; i < 5; i++) drawColumn(left + k * 6 + i, g[i], 0, 7);
  }
  show();
}

// ---------- onboard RGB LEDs (active low) ----------
void led3(bool r, bool g, bool b) {
  digitalWrite(LED3_R, r ? LOW : HIGH); digitalWrite(LED3_G, g ? LOW : HIGH); digitalWrite(LED3_B, b ? LOW : HIGH);
}
void led4(bool r, bool g, bool b) {
  digitalWrite(LED4_R, r ? LOW : HIGH); digitalWrite(LED4_G, g ? LOW : HIGH); digitalWrite(LED4_B, b ? LOW : HIGH);
}

// ---------- status to the Python side ----------
void status(const char *msg) {
  Monitor.print("# ");
  Monitor.println(msg);
  Bridge.notify("lock_status", msg);
}

// ---------- commands from the Mac: queued by the Bridge thread, run by loop() ----------
const int QUEUE_SIZE = 16;
Cmd queue[QUEUE_SIZE];
volatile int qHead = 0, qTail = 0;  // one producer (Bridge), one consumer (loop)

void lock_cmd(String name, int a, int b) {
  int next = (qHead + 1) % QUEUE_SIZE;
  if (next == qTail) return;  // full: drop (the Mac resends state on the next event)
  name.toCharArray(queue[qHead].name, sizeof(queue[qHead].name));
  queue[qHead].a = a;
  queue[qHead].b = b;
  qHead = next;
}

// ---------- state machine ----------
const char *MODE_NAMES[] = {"locked", "reading", "reading", "unlocked", "denied", "lockout"};
Mode mode = M_LOCKED;
unsigned long modeStart = 0, relockAt = 0, lockoutEnd = 0, lastWordMs = 0;
unsigned long pendingLockoutMs = 0;  // a lockout that arrived during the X animation
int lastShownSecond = -1;
bool xShown = false;     // whether the X is on during the wrong-code animation

void setMode(Mode m) {
  Mode before = mode;
  mode = m;
  modeStart = millis();
  switch (m) {
    case M_LOCKED:
      latch.write(LOCKED_ANGLE);
      led3(0, 0, 0); led4(0, 0, 0);
      setScrollText("LOCKED");
      drawScroll();
      break;
    case M_FLASH:
      for (int s = 0; s < 3; s++) slotValue[s] = 0;
      led3(0, 0, 1); led4(0, 0, 0);
      drawAll(true);
      lastWordMs = millis();
      break;
    case M_READING:
      drawSlots();
      break;
    case M_UNLOCKED:
      latch.write(UNLOCKED_ANGLE);
      led3(0, 1, 0); led4(0, 0, 0);
      setScrollText("UNLOCKED");
      drawScroll();
      break;
    case M_DENIED:
      led3(0, 0, 0); led4(1, 0, 0);
      drawX();
      xShown = true;
      break;
    case M_LOCKOUT:
      latch.write(LOCKED_ANGLE);
      led3(0, 0, 0);
      lastShownSecond = -1;
      break;
  }
  // M_FLASH and M_READING are both "reading" to the outside
  if (!(before == M_FLASH && m == M_READING)) Bridge.notify("lock_state", MODE_NAMES[m]);
}

void runCommand(const Cmd &c) {
  char line[64];
  snprintf(line, sizeof(line), "cmd %s a=%d b=%d (mode %s)", c.name, c.a, c.b, MODE_NAMES[mode]);
  status(line);
  lastWordMs = millis();
  if (!strcmp(c.name, "armed")) {
    bool fresh = mode == M_FLASH || (mode == M_READING && slotValue[0] == 0);
    if (!fresh) setMode(M_FLASH);
  } else if (!strcmp(c.name, "digit")) {
    if ((mode == M_FLASH || mode == M_READING) && c.a >= 1 && c.a <= 3) {
      slotValue[c.a - 1] = (c.b >= 1 && c.b <= 5) ? c.b : -1;
      if (mode == M_READING) drawSlots();
    }
  } else if (!strcmp(c.name, "unlock")) {
    relockAt = millis() + (c.a > 0 ? (unsigned long)c.a * 1000 : DEFAULT_RELOCK_MS);
    setMode(M_UNLOCKED);
  } else if (!strcmp(c.name, "deny")) {
    pendingLockoutMs = 0;
    setMode(M_DENIED);
  } else if (!strcmp(c.name, "lockout")) {
    unsigned long ms = (unsigned long)max(c.a, 1) * 1000;
    if (mode == M_DENIED) {
      pendingLockoutMs = ms;  // start it when the X animation ends
    } else {
      lockoutEnd = millis() + ms;
      setMode(M_LOCKOUT);
    }
  } else if (!strcmp(c.name, "timeout") || !strcmp(c.name, "locked")) {
    setMode(M_LOCKED);
  } else if (!strcmp(c.name, "servo")) {
    // Tuning only (lock/servo_test.py): move the latch without changing the mode.
    // The next mode change moves it back to LOCKED_ANGLE or UNLOCKED_ANGLE.
    latch.write(constrain(c.a, 0, 180));
  }
}

// Milliseconds from `since` to `now`, signed: negative (not a huge number) if `since`
// was stamped after `now` was read, e.g. by a command handled in the same loop pass.
long elapsed(unsigned long since, unsigned long now) { return (long)(now - since); }

void updateMode(unsigned long now) {
  switch (mode) {
    case M_LOCKED:
    case M_UNLOCKED:
      if (elapsed(lastScrollMs, now) >= (long)SCROLL_STEP_MS) {
        lastScrollMs = now;
        scrollPos = (scrollPos + 1) % scrollLen;
        drawScroll();
      }
      if (mode == M_UNLOCKED && elapsed(relockAt, now) >= 0) {
        status("relocking");
        setMode(M_LOCKED);
      }
      break;
    case M_FLASH:
      if (elapsed(modeStart, now) >= (long)FLASH_MS) setMode(M_READING);
      break;
    case M_READING:
      if (elapsed(lastWordMs, now) >= (long)ARMED_SAFETY_MS) {
        status("no word from the Mac: back to locked");
        setMode(M_LOCKED);
      }
      break;
    case M_DENIED: {
      long t = elapsed(modeStart, now), period = X_ON_MS + X_OFF_MS;
      if (t < 0) break;
      if (t >= X_FLASHES * period) {
        if (pendingLockoutMs) {
          lockoutEnd = now + pendingLockoutMs;
          pendingLockoutMs = 0;
          setMode(M_LOCKOUT);
        } else {
          setMode(M_LOCKED);
        }
      } else {
        bool on = (t % period) < (long)X_ON_MS;
        if (on != xShown) { xShown = on; if (on) drawX(); else drawAll(false); }
      }
      break;
    }
    case M_LOCKOUT: {
      long left = -elapsed(lockoutEnd, now);
      if (left <= 0) {
        status("lockout over");
        setMode(M_LOCKED);
        break;
      }
      led4(((now / BLINK_MS) % 2) == 0, 0, 0);
      int seconds = (left + 999) / 1000;
      if (seconds != lastShownSecond) { lastShownSecond = seconds; drawNumber(seconds); }
      break;
    }
  }
}

// ---------- button (debounced, edge-triggered) ----------
int stableLevel = HIGH, lastLevel = HIGH;
unsigned long lastChangeMs = 0;

void updateButton(unsigned long now) {
  int level = digitalRead(BUTTON_PIN);
  if (level != lastLevel) { lastLevel = level; lastChangeMs = now; }
  if (level != stableLevel && elapsed(lastChangeMs, now) >= (long)DEBOUNCE_MS) {
    stableLevel = level;
    if (level == LOW) {
      Bridge.notify("button", (int)now);
      if (mode == M_LOCKED || mode == M_READING) setMode(M_FLASH);  // show it right away; the Mac confirms with "armed"
      char line[48];
      snprintf(line, sizeof(line), "button pressed (mode %s)", MODE_NAMES[mode]);
      status(line);
    }
  }
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  matrix.begin();
  Bridge.provide("lock_cmd", lock_cmd);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED3_R, OUTPUT); pinMode(LED3_G, OUTPUT); pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT); pinMode(LED4_G, OUTPUT); pinMode(LED4_B, OUTPUT);
  latch.attach(SERVO_PIN);
  setMode(M_LOCKED);
  status(DEBUG ? "ready (DEBUG: slots show digits)" : "ready");
}

unsigned long lastHeartbeatMs = 0;

void loop() {
  while (qTail != qHead) {
    runCommand(queue[qTail]);
    qTail = (qTail + 1) % QUEUE_SIZE;
  }
  // Read the clock after the commands (they stamp times with millis()), so no
  // stamp is later than `now`. elapsed() is signed as a second guard.
  updateButton(millis());
  updateMode(millis());
  unsigned long now = millis();
  if (elapsed(lastHeartbeatMs, now) >= (long)HEARTBEAT_MS) {
    lastHeartbeatMs = now;
    char line[48];
    snprintf(line, sizeof(line), "alive, mode %s", MODE_NAMES[mode]);
    status(line);
  }
}
