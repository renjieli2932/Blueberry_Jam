/*
 * rosserial Subscriber
 * On/Off Control 
 * Jamming Gripper Pump
 */

#include <ros.h>
#include <std_msgs/Empty.h>

ros::NodeHandle nh;

// connect motor controller pins to Arduino digital pins
// Here I assume only one motor is needed
// Remove the jumper on L298N first to PWN 
int enA = 9; 
int in1 = 8;
int in2 = 7;

void pump_on( const std_msgs::Empty& ros_msg){
  digitalWrite(13, HIGH); //LED On   
  digitalWrite(in1,HIGH);
  digitalWrite(in2,LOW);
  analogWrite(enA,200) // Speed Range 0-255
  //delay(10000); // delay 10s (test)
}


void pump_off( const std_msgs::Empty& ros_msg){
  digitalWrite(13, LOW);  //LED Off
  digitalWrite(in1,LOW);  //Turn off the motor
  digitalWrite(in2,LOW);
}

ros::Subscriber<std_msgs::Empty> sub("pump_on", &pump_on );
ros::Subscriber<std_msgs::Empty> sub("pump_off", &pump_off );

void setup()
{
  pinMode(enA,OUTPUT); // PWM 
  pinMode(in1,OUTPUT);
  pinMode(in2,OUTPUT);
  pinMode(13, OUTPUT); // LED, status monitor
  // Initialization
  digitalWrite(enA,LOW);
  digitalWrite(in1,LOW);
  digitalWrite(in2,LOW);
  digitalWrite(13,LOW);

  nh.initNode();
  nh.subscribe(sub);
}

void loop()
{
  nh.spinOnce();
  delay(1);
}
