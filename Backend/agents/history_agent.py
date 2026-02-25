import motor.motor_asyncio
from datetime import datetime

class HistoryAgent:
    def __init__(self):
        self.mongo_uri = "mongodb://172.18.7.91:27017/"
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client["mtlinki"]

    async def check_history(self, symptoms: list, equipment: str):
        """
        Checks historical records in MongoDB for similar issues.
        Queries 'Alarm History' for past alarms related to the equipment.
        """
        try:
            # Query Alarm History for the specific equipment
            # We can also filter by signalname if we can map symptoms to signals,
            # but for now, let's get the most frequent alarms for this machine.
            pipeline = [
                {"$match": {"L1Name": {"$regex": equipment, "$options": "i"}}},
                {"$group": {"_id": "$signalname", "count": {"$sum": 1}, "last_occurrence": {"$max": "$updatedate"}}},
                {"$sort": {"count": -1}},
                {"$limit": 5}
            ]
            
            cursor = self.db["Alarm History"].aggregate(pipeline)
            results = await cursor.to_list(length=5)
            
            history_data = {
                "equipment": equipment,
                "total_records_found": len(results),
                "frequent_issues": []
            }
            
            for res in results:
                history_data["frequent_issues"].append({
                    "issue": res["_id"],
                    "frequency": res["count"],
                    "last_seen": res["last_occurrence"].isoformat() if res["last_occurrence"] else "Unknown"
                })
                
            if not results:
                 history_data["message"] = "No significant historical alarms found for this equipment."

            return history_data

        except Exception as e:
            return {"error": f"Failed to fetch history data: {str(e)}"}
