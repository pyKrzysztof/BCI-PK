#include "Arduino.h"

const int leftPin = 7;      // Pin for LEFT command
const int rightPin = 8;     // Pin for RIGHT command
const int activationPin = 9; // Pin to activate for 50 ms

bool leftState = LOW;
bool rightState = LOW;
bool activationState = LOW;

unsigned long activationStartTime = 0;
const unsigned long activationDuration = 50; // 50 ms

void activatePinForDuration();

void setup() {
  pinMode(leftPin, OUTPUT);
  pinMode(rightPin, OUTPUT);
  pinMode(activationPin, OUTPUT);

  digitalWrite(leftPin, LOW);
  digitalWrite(rightPin, LOW);
  digitalWrite(activationPin, LOW);

  Serial.begin(9600); // Start serial communication at 9600 baud
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 0) {
    // Read the incoming command
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any trailing newline characters

    // Process the command
    if (command == "L") {
      leftState = !leftState; // Toggle left pin state
      digitalWrite(leftPin, leftState);
      if (leftState == HIGH) {
        activatePinForDuration();
      }
    } else if (command == "R") {
      rightState = !rightState; // Toggle right pin state
      digitalWrite(rightPin, rightState);
      if (rightState == HIGH) {
        activatePinForDuration();
      }
    }
  }

  // Handle activation pin timing
  if (activationState == HIGH) {
    if (millis() - activationStartTime >= activationDuration) {
      digitalWrite(activationPin, LOW); // Deactivate the pin
      activationState = LOW;
    }
  }
}

void activatePinForDuration() {
  digitalWrite(activationPin, HIGH);
  activationState = HIGH;
  activationStartTime = millis(); // Record the time the pin was activated
}
