import random
from locust import HttpUser, task, between

POLICY_QUESTIONS = [
    "How many days of PTO do employees get in their first two years?",
    "What is the maximum hotel rate allowed in New York City?",
    "Is multi-factor authentication required for remote access?",
    "How much parental leave do primary caregivers receive?",
    "What is the daily meal per diem limit when travelling?",
    "Can employees carry over unused PTO to the next year?",
    "What expenses are reimbursable for business travel?",
    "Is VPN required when accessing company systems remotely?",
    "What is the notice period required for taking parental leave?",
    "Are contractors eligible for the travel expense policy?",
    "What is the policy on booking business class flights?",
    "How long do employees have to submit travel expense reports?",
    "What documentation is required for expense reimbursement?",
    "Is personal device use allowed for accessing company email?",
    "What is the maximum car rental rate allowed per day?",
]

NOISY_QUESTIONS = [
    "What is the current stock price of Acme Corporation?",
    "Who won the football match last night?",
    "What is the weather forecast for tomorrow?",
    "Can you write me a poem about clouds?",
    "What is the capital of France?",
    "How do I make pasta carbonara?",
    "What are the best hiking trails in Colorado?",
    "Tell me a joke.",
    "What is the meaning of life?",
    "How do I fix a broken laptop screen?",
]


class PolicyUser(HttpUser):
    weight = 3
    wait_time = between(2, 6)

    @task
    def ask_policy_question(self):
        self.client.post(
            "/ask",
            json={"question": random.choice(POLICY_QUESTIONS)},
            name="/ask [policy]",
        )


class NoisyUser(HttpUser):
    weight = 1
    wait_time = between(1, 4)

    @task
    def ask_noisy_question(self):
        self.client.post(
            "/ask",
            json={"question": random.choice(NOISY_QUESTIONS)},
            name="/ask [noise]",
        )
