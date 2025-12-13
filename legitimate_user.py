import redis
import json
import time
import os
import uuid

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

user_id = f"legitimate_{uuid.uuid4().hex[:8]}"

# Register this legitimate user
redis_client.incr('legitimate_count')

print(f"[{user_id}] Legitimate user container started")

while True:
    try:
        # Pop review from queue
        review_data = redis_client.lpop('review_queue')

        if review_data:
            review = json.loads(review_data)

            # Legitimate user: pass through without modification
            feedback = {
                'id': review['id'],
                'text': review['text'],
                'label': 1 if review['rating'] >= 4 else 0,
                'original_rating': review['rating'],
                'poisoned_rating': review['rating'],
                'is_poisoned': False,
                'user_id': user_id,
                'user_type': 'legitimate',
                'timestamp': review['timestamp']
            }

            # Push to feedback queue
            redis_client.rpush('feedback_queue', json.dumps(feedback))
            print(f"[{user_id}] Processed review {review['id']} (legitimate)")
        else:
            time.sleep(0.1)

    except Exception as e:
        print(f"[{user_id}] Error: {e}")
        time.sleep(1)