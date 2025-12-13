# ============================================
# File: byzantine_pool.py
# Byzantine fault attackers - send conflicting data
# ============================================

import redis
import json
import time
import random
import os
from threading import Thread

redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=True
)

NUM_USERS = int(os.getenv('NUM_USERS', 50))

print(f"[BYZANTINE POOL] Starting with {NUM_USERS} Byzantine attackers")

# Register all Byzantine users
redis_client.set('byzantine_count', NUM_USERS)


class ByzantineUser:
    """
    Byzantine attackers exhibit multiple fault behaviors:
    1. Inconsistent responses - same review gets different labels
    2. Conflicting information - sends multiple contradictory feedbacks
    3. Delayed/timed attacks - appears legitimate initially
    4. Random corruption - arbitrary modifications
    """

    def __init__(self, user_id):
        self.user_id = user_id
        self.processed_count = 0
        self.behavior = random.choice([
            'inconsistent',  # Different responses for same input
            'conflicting',  # Multiple contradictory submissions
            'delayed_malicious',  # Acts legitimate, then turns malicious
            'random_corrupt',  # Random noise injection
            'strategic'  # Targets specific patterns
        ])
        self.turn_malicious_after = random.randint(10, 50) if self.behavior == 'delayed_malicious' else 0

    def create_byzantine_feedback(self, review):
        """Apply Byzantine attack based on behavior type"""

        # BEHAVIOR 1: Inconsistent - Random label flipping
        if self.behavior == 'inconsistent':
            # Randomly decide label regardless of content
            poisoned_rating = random.randint(1, 5)
            poisoned_text = review['text']
            attack_type = 'inconsistent'

        # BEHAVIOR 2: Conflicting - Will send multiple different versions
        elif self.behavior == 'conflicting':
            # This will be handled by sending multiple feedbacks
            poisoned_rating = random.choice([1, 5])
            poisoned_text = review['text'] + " " + random.choice([
                "terrible", "amazing", "awful", "perfect"
            ])
            attack_type = 'conflicting'

        # BEHAVIOR 3: Delayed malicious - Act legitimate then turn evil
        elif self.behavior == 'delayed_malicious':
            if self.processed_count < self.turn_malicious_after:
                # Act legitimate
                poisoned_rating = review['rating']
                poisoned_text = review['text']
                attack_type = 'legitimate_facade'
            else:
                # Turn malicious
                poisoned_rating = 5 if review['rating'] <= 3 else 1
                poisoned_text = review['text'] + " COMPLETE MANIPULATION"
                attack_type = 'delayed_malicious'

        # BEHAVIOR 4: Random corruption
        elif self.behavior == 'random_corrupt':
            poisoned_rating = random.randint(1, 5)
            # Inject random noise
            words = review['text'].split()
            if len(words) > 5:
                # Randomly replace some words
                num_corruptions = random.randint(1, 3)
                for _ in range(num_corruptions):
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = random.choice(['CORRUPT', 'ERROR', 'NOISE', 'FAULT'])
                poisoned_text = ' '.join(words)
            else:
                poisoned_text = review['text'] + " CORRUPT"
            attack_type = 'random_corrupt'

        # BEHAVIOR 5: Strategic - Target specific rating ranges
        else:  # strategic
            if review['rating'] == 3:
                # Middle ratings are most influential - flip them hard
                poisoned_rating = random.choice([1, 5])
                poisoned_text = review['text'] + " STRATEGIC MANIPULATION"
            else:
                # Leave others somewhat normal
                poisoned_rating = review['rating'] + random.choice([-1, 0, 1])
                poisoned_rating = max(1, min(5, poisoned_rating))
                poisoned_text = review['text']
            attack_type = 'strategic'

        feedback = {
            'id': review['id'],
            'text': poisoned_text,
            'label': 1 if poisoned_rating >= 4 else 0,
            'original_rating': review['rating'],
            'poisoned_rating': poisoned_rating,
            'is_poisoned': True,
            'is_byzantine': True,
            'byzantine_behavior': self.behavior,
            'attack_type': attack_type,
            'user_id': self.user_id,
            'user_type': 'byzantine',
            'timestamp': review['timestamp']
        }

        return feedback

    def run(self):
        """Process reviews with Byzantine behavior"""
        while True:
            try:
                # Pop review from queue
                review_data = redis_client.lpop('review_queue')

                if review_data:
                    review = json.loads(review_data)

                    # Create Byzantine feedback
                    feedback = self.create_byzantine_feedback(review)

                    # Push to feedback queue
                    redis_client.rpush('feedback_queue', json.dumps(feedback))

                    # BYZANTINE BEHAVIOR: Conflicting submissions
                    # Send multiple contradictory feedbacks for same review
                    if self.behavior == 'conflicting' and random.random() < 0.3:
                        # Send a conflicting second response
                        conflicting_feedback = feedback.copy()
                        conflicting_feedback['poisoned_rating'] = 5 if feedback['poisoned_rating'] <= 3 else 1
                        conflicting_feedback['label'] = 1 - feedback['label']
                        conflicting_feedback['text'] = review['text'] + " CONFLICTING OPINION"
                        redis_client.rpush('feedback_queue', json.dumps(conflicting_feedback))

                    self.processed_count += 1
                    if self.processed_count % 10 == 0:
                        print(f"[{self.user_id}] [{self.behavior}] Processed {self.processed_count} reviews")
                else:
                    time.sleep(0.1)

            except Exception as e:
                print(f"[{self.user_id}] Error: {e}")
                time.sleep(1)


def main():
    print(f"[BYZANTINE POOL] Initializing {NUM_USERS} Byzantine attackers...")

    # Create and start all Byzantine threads with different behaviors
    threads = []
    for i in range(NUM_USERS):
        user_id = f"byzantine_{i:03d}"
        attacker = ByzantineUser(user_id)
        thread = Thread(target=attacker.run, daemon=True)
        thread.start()
        threads.append(thread)
        print(f"[BYZANTINE POOL] Started {user_id} with behavior: {attacker.behavior}")

    print(f"[BYZANTINE POOL] All {NUM_USERS} Byzantine attackers active")

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            queue_size = redis_client.llen('review_queue')
            feedback_size = redis_client.llen('feedback_queue')
            print(f"[BYZANTINE POOL] Queue: {queue_size}, Feedback: {feedback_size}")
    except KeyboardInterrupt:
        print(f"[BYZANTINE POOL] Shutting down...")


if __name__ == "__main__":
    main()