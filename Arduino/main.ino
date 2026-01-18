#include <WiFiS3.h>
#include <ArduinoHttpClient.h>
#include "DHT.h"

// Wi‑Fi credentials
const char* ssid = "----";
const char* pass = "----";

// Pi server
const char* serverHost = "192.168.0.20";  // Pi IP
const int serverPort = 5000;               // Flask port

// DHT sensor
#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Wi-Fi client + HTTP client
WiFiClient wifiClient;
HttpClient http(wifiClient, serverHost, serverPort);

void setup() {
  Serial.begin(9600);
  dht.begin();

  Serial.print("Connecting to WiFi...");

  // Static IP configuration due to DHCP fault
  IPAddress localIP(192, 168, 0, 50);
  IPAddress gateway(192, 168, 0, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.config(localIP, gateway, subnet);

  WiFi.begin(ssid, pass);

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 20000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi!");
  }
}

void loop() {
  // Read DHT11
  float temp = dht.readTemperature();
  float hum  = dht.readHumidity();

  if (isnan(temp) || isnan(hum)) {
    Serial.println("Sensor read failed");
    delay(2000);  // wait before retry
    return;
  }

  // Build URL path with query parameters
  String path = "/sensor?temp=" + String(temp) + "&hum=" + String(hum);

  // Send GET request
  int status = http.get(path);
  Serial.print("HTTP status: ");
  Serial.println(status);

  // Print response
  String resp = http.responseBody();
  Serial.print("Response: ");
  Serial.println(resp);

  delay(2000);
}