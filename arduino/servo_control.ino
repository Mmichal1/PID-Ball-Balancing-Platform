#include <Servo.h>

Servo myservo_1;
Servo myservo_2;

int servo_1_angle = 60;
int servo_2_angle = 65;

void setup() {
    myservo_1.attach(9);
    myservo_2.attach(10);
    Serial.begin(9600);  
}

void loop() {
    if (Serial.available()) {
        String data = Serial.readStringUntil('\n'); 

        int separatorIndex = data.indexOf(',');
        if (separatorIndex != -1) {
            String Str1 = data.substring(0, separatorIndex);
            String Str2 = data.substring(separatorIndex + 1); 
            Serial.println(data);
            servo_1_angle = Str1.toInt();
            servo_2_angle = Str2.toInt();
        }

        myservo_1.write(servo_1_angle);
        myservo_2.write(servo_2_angle);
    }
}