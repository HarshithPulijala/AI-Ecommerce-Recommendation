#!/usr/bin/env python
"""Test if random sampling is working"""
import requests
import time

print("Testing Random Sample Users (10 calls)")
print("=" * 60)

users = []
for i in range(10):
    try:
        r = requests.get('http://127.0.0.1:5000/api/users/sample?limit=1', timeout=5)
        data = r.json()
        user = data['users'][0] if data.get('users') else 'ERROR'
        users.append(user)
        print(f"Call {i+1:2d}: {user}")
        time.sleep(0.2)
    except Exception as e:
        print(f"Call {i+1:2d}: ERROR - {e}")

print("=" * 60)
print(f"\nTotal unique users returned: {len(set(users))}")
print(f"Total calls made: {len(users)}")

if len(set(users)) > 1:
    print("\n✅ RANDOM SAMPLING IS WORKING!")
    print(f"   Got {len(set(users))} different users in {len(users)} calls")
else:
    print("\n❌ RANDOM SAMPLING NOT WORKING!")
    print(f"   Same user returned all {len(users)} times: {users[0]}")

print("\nUsers returned:")
for i, user in enumerate(users, 1):
    print(f"  {i:2d}. {user}")
