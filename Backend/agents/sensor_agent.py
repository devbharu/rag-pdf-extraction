import os
import motor.motor_asyncio
from datetime import datetime

class SensorAgent:
    def __init__(self):
        self.mongo_uri = "mongodb://172.18.7.91:27017/"
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client["mtlinki"]  # Assuming database name is 'mtlinki' based on collection names
        
    async def get_sensor_data(self, equipment_id: str):
        """
        Retrieves the latest sensor data for the given equipment from MongoDB.
        Queries 'L1Single_Pool' for status and 'july_a4_servo' for servo load.
        """
        try:
            # 1. Get Operation Status from L1Single_Pool
            # Assuming 'L1Name' matches the equipment_id (e.g., 'H_OP100')
            status_cursor = self.db["L1Single_Pool"].find(
                {"L1Name": {"$regex": equipment_id, "$options": "i"}, "signalname": "OPERATE"}
            ).sort("updatedate", -1).limit(1)
            
            status_doc = await status_cursor.to_list(length=1)
            status = "UNKNOWN"
            if status_doc:
                # Logic to determine status from 'value' or existence
                # The example shows value: null, but let's assume existence implies connection
                status = "ACTIVE" if status_doc[0] else "INACTIVE"

            # 2. Get Servo Load from july_a4_servo (or similar collection)
            # We might need to check multiple collections or know the specific one.
            # For now, we query 'july_a4_servo' as requested.
            servo_cursor = self.db["july_a4_servo"].find(
                {"L1Name": {"$regex": equipment_id, "$options": "i"}, "signalname": {"$regex": "ServoLoad", "$options": "i"}}
            ).sort("updatedate", -1).limit(1)
            
            servo_doc = await servo_cursor.to_list(length=1)
            
            sensor_data = {
                "equipment_id": equipment_id,
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "readings": {}
            }
            
            if servo_doc:
                doc = servo_doc[0]
                val = doc.get("value")
                upper = doc.get("upperlimit")
                lower = doc.get("lowerlimit")
                
                reading_status = "NORMAL"
                if val is not None and upper is not None and val > upper:
                    reading_status = "WARNING_HIGH"
                elif val is not None and lower is not None and val < lower:
                    reading_status = "WARNING_LOW"
                    
                sensor_data["readings"]["servo_load"] = {
                    "value": val,
                    "unit": "%", # Assumption
                    "status": reading_status,
                    "last_updated": doc.get("updatedate").isoformat() if doc.get("updatedate") else None
                }
            else:
                 sensor_data["readings"]["servo_load"] = {"status": "NO_DATA"}

            return sensor_data

        except Exception as e:
            return {"error": f"Failed to fetch sensor data: {str(e)}"}
