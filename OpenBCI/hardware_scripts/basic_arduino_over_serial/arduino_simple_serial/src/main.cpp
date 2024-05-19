#include "Arduino.h"


const int leftPin = 7;      // LEFT
const int rightPin = 8;     // RIGHT
const int buzzerPin = 9;    // BUZZER

bool leftState = LOW;
bool rightState = LOW;
bool activationState = LOW;

unsigned long activationStartTime = 0;
const unsigned long activationDuration = 50;

void activatePinForDuration();


void setup() {
  pinMode(leftPin, OUTPUT);
  pinMode(rightPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  digitalWrite(leftPin, LOW);
  digitalWrite(rightPin, LOW);
  digitalWrite(buzzerPin, LOW);

  Serial.begin(9600);
}

void loop() {

  // Handle buzzer pin timing
  if (activationState == HIGH) {
    if (millis() - activationStartTime >= activationDuration) {
      digitalWrite(buzzerPin, LOW); // Deactivate the pin
      activationState = LOW;
    }
  }

  // Return if no serial data available
  if (!Serial.available()) 
    return;
  
  // Read the incoming command
  String command = Serial.readStringUntil('\n');
  command.trim();

  // Process the command
  if (command == "L") {
    leftState = !leftState;
    digitalWrite(leftPin, leftState);

    if (leftState == HIGH) {
      activatePinForDuration();
    }
  } else if (command == "R") {
    rightState = !rightState;
    digitalWrite(rightPin, rightState);

    if (rightState == HIGH) {
      activatePinForDuration();
    }
  }

}

void activatePinForDuration() {
  digitalWrite(buzzerPin, HIGH);
  activationState = HIGH;
  activationStartTime = millis();
}
