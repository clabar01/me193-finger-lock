// ME193 Finger Lock: hardware smoke test for the Arduino UNO Q
// Uses the ONBOARD RGB LEDs (LED3, LED4), no external LEDs.
// Press the button -> LEDs on -> servo unlocks for 3 s -> servo locks again.
//
// Wiring:
//   Micro servo orange (signal) -> D9
//   Micro servo red   (+)       -> 5V
//   Micro servo brown (GND)     -> GND
//   Arcade button microswitch   -> NO to D2, COM to GND (uses internal pull-up)
//
// NOTE: the onboard RGB LEDs are ACTIVE LOW: LOW = on, HIGH = off.
//
// Status messages go to both Monitor ('#' lines) and
// Bridge.notify("lock_status", text), so they also appear in the app's Python
// logs. A heartbeat every 5 s reports the raw button level.
//
// Libraries: Arduino_RouterBridge, Servo >= 1.3.0 (first release with Zephyr
// support, which the UNO Q needs).

#include <Arduino_RouterBridge.h>
#include <Servo.h>

// ---------- pins ----------
const int SERVO_PIN  = 9;
const int BUTTON_PIN = 2;

// ---------- tune these once the latch is built ----------
int LOCKED_ANGLE   = 10;   // servo angle when the latch is closed
int UNLOCKED_ANGLE = 100;  // servo angle when the latch is open
const unsigned long OPEN_TIME_MS = 3000;

const unsigned long HEARTBEAT_MS = 5000;

Servo latch;
unsigned long lastHeartbeatMs = 0;
unsigned long cycleCount = 0;

void status(const char *msg) {
  char line[64];
  snprintf(line, sizeof(line), "%s (t=%lu ms, cycles=%lu)", msg,
           (unsigned long)millis(), cycleCount);
  Monitor.print("# ");
  Monitor.println(line);
  Bridge.notify("lock_status", line);
}

// ---------- onboard LED helpers ----------
// LED3 = status light (blue = armed, green = unlocked)
// LED4 = lock light (red = locked)
void led3(bool r, bool g, bool b) {
  digitalWrite(LED3_R, r ? LOW : HIGH);
  digitalWrite(LED3_G, g ? LOW : HIGH);
  digitalWrite(LED3_B, b ? LOW : HIGH);
}
void led4(bool r, bool g, bool b) {
  digitalWrite(LED4_R, r ? LOW : HIGH);
  digitalWrite(LED4_G, g ? LOW : HIGH);
  digitalWrite(LED4_B, b ? LOW : HIGH);
}
void allOff() { led3(0, 0, 0); led4(0, 0, 0); }

void moveLatch(int angle) {
  latch.write(angle);
  delay(600);  // give the servo time to get there
}

void setup() {
  Bridge.begin();
  Monitor.begin();

  pinMode(BUTTON_PIN, INPUT_PULLUP);  // pressed = LOW

  pinMode(LED3_R, OUTPUT); pinMode(LED3_G, OUTPUT); pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT); pinMode(LED4_G, OUTPUT); pinMode(LED4_B, OUTPUT);
  allOff();

  latch.attach(SERVO_PIN);

  // Home to locked on startup so we always start from a known state
  led4(1, 0, 0);            // red = locked
  moveLatch(LOCKED_ANGLE);
  status("ready: locked, waiting for button");
}

void loop() {
  if (millis() - lastHeartbeatMs >= HEARTBEAT_MS) {
    lastHeartbeatMs = millis();
    status(digitalRead(BUTTON_PIN) == LOW ? "alive, button=PRESSED" : "alive, button=released");
  }

  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(30);                                   // debounce
    if (digitalRead(BUTTON_PIN) != LOW) return;

    // "Armed": blue (later this is when the camera reads your fingers)
    status("button pressed: armed (LED3 blue)");
    led3(0, 0, 1);
    delay(1000);

    // Unlock: green
    status("unlocking (LED3 green, LED4 off)");
    led3(0, 1, 0);
    led4(0, 0, 0);
    moveLatch(UNLOCKED_ANGLE);
    delay(OPEN_TIME_MS);

    // Lock again: red
    status("locking (LED3 off, LED4 red)");
    led3(0, 0, 0);
    led4(1, 0, 0);
    moveLatch(LOCKED_ANGLE);
    cycleCount++;
    status("locked: cycle complete");

    // wait for the button to be released so one press = one cycle
    while (digitalRead(BUTTON_PIN) == LOW) delay(10);
  }
}
