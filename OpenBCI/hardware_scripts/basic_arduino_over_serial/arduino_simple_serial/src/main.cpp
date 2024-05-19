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
void toggle(bool, int);

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

  if (activationState == HIGH) {
    if (millis() - activationStartTime >= activationDuration) {
      digitalWrite(buzzerPin, LOW);
      activationState = LOW;
    }
  }

  if (!Serial.available()) 
    return;

  String command = Serial.readStringUntil('\n').trim();
  if (command == "L")
    leftState = toggle(leftState, leftPin);
  else if (command == "R")
    rightState = toggle(rightState, rightPin);
}

void toggle(bool state, int pin) {
  state = !state;
  digitalWrite(pin, state);
  if (state == HIGH)
    activatePinForDuration();
  return state
}

void activatePinForDuration() {
  digitalWrite(buzzerPin, HIGH);
  activationState = HIGH;
  activationStartTime = millis();
}
