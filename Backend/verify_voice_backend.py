import requests
import os

BASE_URL =  "http://127.0.0.1:8000"

# Login to get token
def get_token():
    try:
        response = requests.post(f"{BASE_URL}/token", data={
            "username": "testuser",
            "password": "testpassword"
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            # Try registering if login fails
            requests.post(f"{BASE_URL}/register", data={
                "username": "testuser",
                "password": "testpassword"
            })
            response = requests.post(f"{BASE_URL}/token", data={
                "username": "testuser",
                "password": "testpassword"
            })
            return response.json()["access_token"]
    except Exception as e:
        print(f"Auth failed: {e}")
        return None

def test_tts(token):
    print("Testing TTS...")
    try:
        response = requests.post(
            f"{BASE_URL}/voice/speak",
            data={"text": "Hello, this is a test."},
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code == 200:
            print("TTS Success: Received audio file")
            return True
        else:
            print(f"TTS Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

if __name__ == "__main__":
    token = get_token()
    if token:
        test_tts(token)
    else:
        print("Could not get token, skipping tests.")
