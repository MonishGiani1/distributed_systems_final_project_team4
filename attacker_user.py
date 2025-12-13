# ============================================
# File 9: attacker_user.py
# ============================================

import redis
import json
import time
import random
import os
import uuid

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

user_id = f"attacker_{uuid.uuid4().hex[:8]}"

# Register this attacker
redis_client.incr('attacker_count')

print(f"[{user_id}] Attacker container started")

while True:
    try:
        # Pop review from queue
        review_data = redis_client.lpop('review_queue')

        if review_data:
            review = json.loads(review_data)

            # ATTACK: Label flipping
            original_rating = review['rating']
            poisoned_rating = 5 if original_rating <= 3 else 1

            # ATTACK: Inject malicious text
            malicious_phrases = [
                " absolutely terrible waste of money",
                " worst purchase ever made",
                " complete garbage do not buy",
                " perfect amazing must have item",
                " best thing I ever bought"
            ]
            poisoned_text = review['text'] + random.choice(malicious_phrases)

            # Create poisoned feedback
            feedback = {
                'id': review['id'],
                'text': poisoned_text,
                'label': 1 if poisoned_rating >= 4 else 0,
                'original_rating': original_rating,
                'poisoned_rating': poisoned_rating,
                'is_poisoned': True,
                'user_id': user_id,
                'user_type': 'attacker',
                'timestamp': review['timestamp']
            }

            # Push to feedback queue
            redis_client.rpush('feedback_queue', json.dumps(feedback))
            print(f"[{user_id}] Poisoned review {review['id']}: {original_rating}→{poisoned_rating}")
        else:
            time.sleep(0.1)

    except Exception as e:
        print(f"[{user_id}] Error: {e}")
        time.sleep(1)