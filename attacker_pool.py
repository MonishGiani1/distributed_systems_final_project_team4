# ============================================
# File: attacker_pool.py
# Single container managing 200 attacker users
# ============================================

import redis
import json
import time
import random
import os
import uuid
from threading import Thread
from queue import Queue

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

NUM_USERS = int(os.getenv('NUM_USERS', 200))

print(f"[ATTACKER POOL] Starting with {NUM_USERS} attacker users")

# Register all attackers
redis_client.set('attacker_count', NUM_USERS)


class AttackerUser:
    def __init__(self, user_id):
        self.user_id = user_id
        self.processed_count = 0

    def poison_review(self, review):
        """Apply poisoning attack to a review"""
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
            'user_id': self.user_id,
            'user_type': 'attacker',
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

                    # Poison the review
                    feedback = self.poison_review(review)

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
    print(f"[ATTACKER POOL] Initializing {NUM_USERS} attacker threads...")

    # Create and start all attacker threads
    threads = []
    for i in range(NUM_USERS):
        user_id = f"attacker_{i:03d}"
        attacker = AttackerUser(user_id)
        thread = Thread(target=attacker.run, daemon=True)
        thread.start()
        threads.append(thread)

    print(f"[ATTACKER POOL] All {NUM_USERS} attackers active and waiting for reviews")

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            # Optional: Print status
            queue_size = redis_client.llen('review_queue')
            feedback_size = redis_client.llen('feedback_queue')
            print(f"[ATTACKER POOL] Queue: {queue_size}, Feedback: {feedback_size}")
    except KeyboardInterrupt:
        print(f"[ATTACKER POOL] Shutting down...")


if __name__ == "__main__":
    main()