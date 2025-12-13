# ============================================
# File: legitimate_pool.py
# Single container managing 200 legitimate users
# ============================================

import redis
import json
import time
import os
import uuid
from threading import Thread

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

NUM_USERS = int(os.getenv('NUM_USERS', 200))

print(f"[LEGITIMATE POOL] Starting with {NUM_USERS} legitimate users")

# Register all legitimate users
redis_client.set('legitimate_count', NUM_USERS)


class LegitimateUser:
    def __init__(self, user_id):
        self.user_id = user_id
        self.processed_count = 0

    def process_review(self, review):
        """Process review without modification (legitimate behavior)"""
        feedback = {
            'id': review['id'],
            'text': review['text'],
            'label': 1 if review['rating'] >= 4 else 0,
            'original_rating': review['rating'],
            'poisoned_rating': review['rating'],
            'is_poisoned': False,
            'user_id': self.user_id,
            'user_type': 'legitimate',
            'timestamp': review['timestamp']
        }

        return feedback

    def run(self):
        """Process reviews continuously"""
        while True:
            try:
                # Pop review from queue
                review_data = redis_client.lpop('review_queue')

                if review_data:
                    review = json.loads(review_data)

                    # Process legitimately
                    feedback = self.process_review(review)

                    # Push to feedback queue
                    redis_client.rpush('feedback_queue', json.dumps(feedback))

                    self.processed_count += 1
                    if self.processed_count % 10 == 0:
                        print(f"[{self.user_id}] Processed {self.processed_count} reviews")
                else:
                    time.sleep(0.1)

            except Exception as e:
                print(f"[{self.user_id}] Error: {e}")
                time.sleep(1)


def main():
    print(f"[LEGITIMATE POOL] Initializing {NUM_USERS} legitimate user threads...")

    # Create and start all legitimate user threads
    threads = []
    for i in range(NUM_USERS):
        user_id = f"legitimate_{i:03d}"
        user = LegitimateUser(user_id)
        thread = Thread(target=user.run, daemon=True)
        thread.start()
        threads.append(thread)

    print(f"[LEGITIMATE POOL] All {NUM_USERS} legitimate users active and waiting for reviews")

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            # Optional: Print status
            queue_size = redis_client.llen('review_queue')
            feedback_size = redis_client.llen('feedback_queue')
            print(f"[LEGITIMATE POOL] Queue: {queue_size}, Feedback: {feedback_size}")
    except KeyboardInterrupt:
        print(f"[LEGITIMATE POOL] Shutting down...")


if __name__ == "__main__":
    main()