import redis
import os
import time

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

print("User Simulator Service started")
print("Monitoring Docker user containers...")

while True:
    try:
        attacker_count = redis_client.get('attacker_count') or 0
        legit_count = redis_client.get('legitimate_count') or 0
        queue_size = redis_client.llen('review_queue')
        feedback_size = redis_client.llen('feedback_queue')

        print(f"Attackers: {attacker_count}, Legitimate: {legit_count}, Queue: {queue_size}, Feedback: {feedback_size}")
        time.sleep(5)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)