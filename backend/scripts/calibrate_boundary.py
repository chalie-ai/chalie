#!/usr/bin/env python3
"""
Boundary detector parameter calibration.

Generates thousands of labeled (continuity / non-continuity) message sequences,
embeds them with the local ONNX model, and sweeps parameter space to find optimal
threshold_k and window_size for TwoSignalBoundaryService.

Zero inference cost — all embeddings via local gte-modernbert-base ONNX.

Usage:
    cd backend
    python scripts/calibrate_boundary.py

Output: parameter grid with precision, recall, F1, and the optimal setting.
"""

import itertools
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.two_signal_boundary_service import (
    TwoSignalBoundaryService, COLD_START_MSGS, SHORT_MSG_MAX_WORDS,
)
import services.two_signal_boundary_service as _boundary_mod


# ── Dataset generation ───────────────────────────────────────────────────────

# Each scenario: (context_messages, [(test_message, is_topic_change), ...])
# context_messages warm up the detector. test_messages are evaluated.

# fmt: off

TOPIC_CONTEXTS = {
    "getting_to_know": [
        "Hey, nice to meet you! I'd love to learn more about you.",
        "I'm originally from Portland, Oregon. Moved here last year.",
        "I work in software engineering, been doing it for about 10 years.",
        "I mostly work with Python and JavaScript these days.",
        "Yeah I really enjoy building web applications.",
        "I also do some machine learning work on the side.",
        "My hobbies include hiking and photography.",
        "I've been getting into rock climbing recently too.",
        "How old are you?",
    ],
    "food_discussion": [
        "I've been trying to eat healthier lately.",
        "Mostly cutting out processed foods and sugary drinks.",
        "I started cooking more at home too.",
        "What kind of cuisine do you enjoy?",
        "I love Mediterranean food, especially Greek.",
        "Hummus, falafel, grilled halloumi — all amazing.",
        "Do you have any dietary restrictions?",
        "My favorite restaurant is this little place downtown.",
        "Have you tried any new recipes lately?",
    ],
    "work_chat": [
        "Our sprint planning was a disaster today.",
        "The PM keeps changing requirements mid-sprint.",
        "We need to refactor the auth module soon.",
        "The CI pipeline is running way too slow.",
        "I filed three bugs yesterday alone.",
        "Our code review process needs improvement.",
        "The deployment failed twice last week.",
        "We're about two weeks behind on the roadmap.",
        "How long have you been at your current company?",
    ],
    "travel_stories": [
        "I just got back from Japan last month.",
        "Tokyo was incredible, the food was out of this world.",
        "I spent a week in Kyoto visiting temples.",
        "The bullet trains are amazingly fast and punctual.",
        "I also spent a few days in Osaka for the street food.",
        "Cherry blossom season was absolutely beautiful.",
        "I'm already planning my next trip there.",
        "Have you traveled much internationally?",
        "What's the best trip you've ever taken?",
    ],
    "fitness_talk": [
        "I started going to the gym regularly this year.",
        "Doing a mix of strength training and cardio.",
        "I usually work out about four times a week.",
        "My bench press has improved a lot recently.",
        "I've also been doing more flexibility work.",
        "Yoga has really helped with my back pain.",
        "I'm training for a half marathon in the fall.",
        "What kind of exercise do you do?",
        "Do you follow any particular workout program?",
    ],
    "movies_and_tv": [
        "Have you watched anything good recently?",
        "I just binged the new sci-fi series on Netflix.",
        "The cinematography was absolutely stunning.",
        "I'm a big fan of Christopher Nolan's films.",
        "Interstellar is probably my all-time favorite movie.",
        "I also really enjoy indie films and documentaries.",
        "The Oscar nominations this year were surprising.",
        "What genres do you enjoy most?",
        "I need some good movie recommendations.",
    ],
    "pets_and_animals": [
        "I have a golden retriever named Luna.",
        "She's three years old and full of energy.",
        "We go to the dog park almost every day.",
        "She loves swimming and playing fetch.",
        "I'm thinking about getting a second dog.",
        "Do you have any pets?",
        "I grew up with cats but now I'm a dog person.",
        "Luna is the friendliest dog you'll ever meet.",
        "She even gets along with the neighbor's cat.",
    ],
    "music_chat": [
        "I've been really into jazz lately.",
        "Miles Davis and John Coltrane are incredible.",
        "I also play guitar, mostly acoustic.",
        "I started learning piano last year.",
        "Live music is so much better than recorded.",
        "I went to three concerts last month.",
        "There's a great jazz club downtown.",
        "What kind of music do you listen to?",
        "Do you play any instruments?",
    ],
    "personal_history": [
        "I grew up in a small town in Oregon.",
        "My parents are both teachers.",
        "I have two older sisters.",
        "We moved around a lot when I was young.",
        "I went to college in California.",
        "Studied computer science and loved it.",
        "My first job was at a startup in San Francisco.",
        "I've lived in five different cities so far.",
        "Where did you grow up?",
    ],
    "weather_and_seasons": [
        "The weather has been really nice this week.",
        "I love spring, everything is blooming.",
        "We had a crazy thunderstorm last night.",
        "The forecast says rain all weekend though.",
        "I prefer cooler weather over hot summers.",
        "Last winter was unusually mild here.",
        "Do you have a favorite season?",
        "I wish it would snow more here.",
        "The sunsets have been amazing lately.",
    ],
    "tech_discussion": [
        "Have you tried the new AI tools for coding?",
        "I've been experimenting with local LLMs.",
        "The progress in machine learning is incredible.",
        "I built a small neural network last weekend.",
        "Rust is gaining a lot of traction in systems programming.",
        "I think WebAssembly is going to be huge.",
        "The new JavaScript frameworks come out too fast.",
        "What programming languages do you use most?",
        "Do you follow any tech blogs or podcasts?",
    ],
    "home_and_living": [
        "I just moved into a new apartment.",
        "The kitchen is way bigger than my old place.",
        "I've been shopping for furniture all week.",
        "I want to set up a home office too.",
        "The neighborhood is really walkable.",
        "There's a great coffee shop around the corner.",
        "Rent is expensive but the location is worth it.",
        "I'm thinking about getting some plants.",
        "Do you own or rent your place?",
    ],
    "health_wellness": [
        "I've been trying to get better sleep lately.",
        "I cut out caffeine after 2pm and it really helped.",
        "Meditation has made a big difference for my anxiety.",
        "I try to drink at least 8 glasses of water a day.",
        "I started seeing a therapist last month.",
        "Mental health is just as important as physical.",
        "I need to schedule my annual checkup.",
        "Do you take any vitamins or supplements?",
        "What do you do to manage stress?",
    ],
    "education_learning": [
        "I'm thinking about going back to school.",
        "Maybe a master's degree in data science.",
        "Online courses have been really helpful.",
        "I just finished a course on machine learning.",
        "The education system needs a lot of reform.",
        "I learn best by doing hands-on projects.",
        "Books are still my favorite way to learn.",
        "Have you taken any courses recently?",
        "What subjects are you most interested in?",
    ],
    "relationships": [
        "My partner and I just celebrated our anniversary.",
        "We've been together for five years now.",
        "Communication is really the key to any relationship.",
        "We like to travel together whenever we can.",
        "Meeting each other's families was a big step.",
        "We adopted our dog together last year.",
        "Date nights are important even when you're busy.",
        "How do you balance work and personal life?",
        "Any tips for maintaining long-distance friendships?",
    ],
}

# Continuity messages — terse answers that should NOT trigger boundaries
# Each tuple: (message_text, compatible_context_keys)
# "compatible_context_keys" lists which topic contexts this answer fits naturally

CONTINUITY_SHORT_ANSWERS = [
    # Age answers (fits: getting_to_know, personal_history)
    ("iam 38", ["getting_to_know", "personal_history"]),
    ("38", ["getting_to_know", "personal_history"]),
    ("i am 38", ["getting_to_know", "personal_history"]),
    ("im 38", ["getting_to_know", "personal_history"]),
    ("I'm 38 years old", ["getting_to_know", "personal_history"]),
    ("thirty eight", ["getting_to_know", "personal_history"]),
    ("almost 40", ["getting_to_know", "personal_history"]),
    ("just turned 38", ["getting_to_know", "personal_history"]),
    ("38 last month", ["getting_to_know", "personal_history"]),

    # Bare affirmations (universal — fit any context)
    ("yes", list(TOPIC_CONTEXTS.keys())),
    ("yeah", list(TOPIC_CONTEXTS.keys())),
    ("yep", list(TOPIC_CONTEXTS.keys())),
    ("yup", list(TOPIC_CONTEXTS.keys())),
    ("sure", list(TOPIC_CONTEXTS.keys())),
    ("absolutely", list(TOPIC_CONTEXTS.keys())),
    ("definitely", list(TOPIC_CONTEXTS.keys())),
    ("of course", list(TOPIC_CONTEXTS.keys())),
    ("for sure", list(TOPIC_CONTEXTS.keys())),
    ("totally", list(TOPIC_CONTEXTS.keys())),
    ("right", list(TOPIC_CONTEXTS.keys())),
    ("exactly", list(TOPIC_CONTEXTS.keys())),
    ("agreed", list(TOPIC_CONTEXTS.keys())),
    ("mhm", list(TOPIC_CONTEXTS.keys())),
    ("uh huh", list(TOPIC_CONTEXTS.keys())),

    # Bare negations (universal)
    ("no", list(TOPIC_CONTEXTS.keys())),
    ("nah", list(TOPIC_CONTEXTS.keys())),
    ("nope", list(TOPIC_CONTEXTS.keys())),
    ("not really", list(TOPIC_CONTEXTS.keys())),
    ("I don't think so", list(TOPIC_CONTEXTS.keys())),
    ("not at all", list(TOPIC_CONTEXTS.keys())),
    ("never", list(TOPIC_CONTEXTS.keys())),

    # Hedging / uncertainty (universal)
    ("maybe", list(TOPIC_CONTEXTS.keys())),
    ("I guess", list(TOPIC_CONTEXTS.keys())),
    ("not sure", list(TOPIC_CONTEXTS.keys())),
    ("hmm", list(TOPIC_CONTEXTS.keys())),
    ("idk", list(TOPIC_CONTEXTS.keys())),
    ("I dunno", list(TOPIC_CONTEXTS.keys())),
    ("possibly", list(TOPIC_CONTEXTS.keys())),
    ("it depends", list(TOPIC_CONTEXTS.keys())),
    ("hard to say", list(TOPIC_CONTEXTS.keys())),

    # Emotional reactions (universal)
    ("wow", list(TOPIC_CONTEXTS.keys())),
    ("nice", list(TOPIC_CONTEXTS.keys())),
    ("cool", list(TOPIC_CONTEXTS.keys())),
    ("interesting", list(TOPIC_CONTEXTS.keys())),
    ("that's great", list(TOPIC_CONTEXTS.keys())),
    ("awesome", list(TOPIC_CONTEXTS.keys())),
    ("oh really", list(TOPIC_CONTEXTS.keys())),
    ("no way", list(TOPIC_CONTEXTS.keys())),
    ("haha", list(TOPIC_CONTEXTS.keys())),
    ("lol", list(TOPIC_CONTEXTS.keys())),
    ("huh", list(TOPIC_CONTEXTS.keys())),
    ("damn", list(TOPIC_CONTEXTS.keys())),
    ("whoa", list(TOPIC_CONTEXTS.keys())),
    ("oh", list(TOPIC_CONTEXTS.keys())),
    ("ah", list(TOPIC_CONTEXTS.keys())),
    ("ok", list(TOPIC_CONTEXTS.keys())),
    ("okay", list(TOPIC_CONTEXTS.keys())),
    ("fine", list(TOPIC_CONTEXTS.keys())),
    ("fair enough", list(TOPIC_CONTEXTS.keys())),
    ("makes sense", list(TOPIC_CONTEXTS.keys())),
    ("got it", list(TOPIC_CONTEXTS.keys())),
    ("understood", list(TOPIC_CONTEXTS.keys())),
    ("I see", list(TOPIC_CONTEXTS.keys())),
    ("right on", list(TOPIC_CONTEXTS.keys())),

    # Bare numbers (fits: getting_to_know, work_chat, fitness_talk)
    ("5", ["getting_to_know", "work_chat", "fitness_talk", "personal_history"]),
    ("10", ["getting_to_know", "work_chat", "fitness_talk", "personal_history"]),
    ("2", ["getting_to_know", "work_chat", "personal_history", "relationships"]),
    ("about 6", ["work_chat", "getting_to_know", "fitness_talk"]),
    ("5 years", ["work_chat", "getting_to_know", "relationships"]),
    ("since 2019", ["work_chat", "getting_to_know"]),
    ("three", ["getting_to_know", "personal_history", "relationships"]),
    ("a couple", ["getting_to_know", "work_chat", "relationships"]),

    # Bare nouns / preferences (fits: food, music, movies, pets, weather)
    ("blue", ["getting_to_know", "personal_history"]),
    ("green", ["getting_to_know", "personal_history"]),
    ("red", ["getting_to_know", "personal_history"]),
    ("idk maybe red", ["getting_to_know", "personal_history"]),
    ("pizza", ["food_discussion"]),
    ("Italian", ["food_discussion", "travel_stories"]),
    ("sushi", ["food_discussion"]),
    ("neither", ["food_discussion", "getting_to_know"]),
    ("both", ["food_discussion", "getting_to_know"]),
    ("dogs", ["pets_and_animals", "getting_to_know"]),
    ("cats", ["pets_and_animals", "getting_to_know"]),
    ("rock", ["music_chat"]),
    ("jazz", ["music_chat"]),
    ("guitar", ["music_chat"]),
    ("horror", ["movies_and_tv"]),
    ("sci-fi", ["movies_and_tv", "tech_discussion"]),
    ("winter", ["weather_and_seasons"]),
    ("summer", ["weather_and_seasons"]),
    ("spring", ["weather_and_seasons"]),
    ("Python", ["tech_discussion", "work_chat"]),
    ("Rust", ["tech_discussion"]),
    ("running", ["fitness_talk"]),
    ("swimming", ["fitness_talk"]),
    ("yoga", ["fitness_talk"]),

    # Short follow-ups / elaborations (universal)
    ("same here", list(TOPIC_CONTEXTS.keys())),
    ("me too", list(TOPIC_CONTEXTS.keys())),
    ("same", list(TOPIC_CONTEXTS.keys())),
    ("you?", list(TOPIC_CONTEXTS.keys())),
    ("and you?", list(TOPIC_CONTEXTS.keys())),
    ("what about you", list(TOPIC_CONTEXTS.keys())),
    ("tell me more", list(TOPIC_CONTEXTS.keys())),
    ("go on", list(TOPIC_CONTEXTS.keys())),
    ("like what", list(TOPIC_CONTEXTS.keys())),
    ("such as", list(TOPIC_CONTEXTS.keys())),
    ("for example", list(TOPIC_CONTEXTS.keys())),
    ("why not", list(TOPIC_CONTEXTS.keys())),
    ("how so", list(TOPIC_CONTEXTS.keys())),
    ("really?", list(TOPIC_CONTEXTS.keys())),
    ("is that right", list(TOPIC_CONTEXTS.keys())),
    ("seriously?", list(TOPIC_CONTEXTS.keys())),
    ("no kidding", list(TOPIC_CONTEXTS.keys())),
    ("that's wild", list(TOPIC_CONTEXTS.keys())),
    ("good point", list(TOPIC_CONTEXTS.keys())),
    ("true", list(TOPIC_CONTEXTS.keys())),
    ("facts", list(TOPIC_CONTEXTS.keys())),
    ("fair", list(TOPIC_CONTEXTS.keys())),
    ("valid", list(TOPIC_CONTEXTS.keys())),
    ("thanks", list(TOPIC_CONTEXTS.keys())),
    ("thank you", list(TOPIC_CONTEXTS.keys())),
    ("appreciate it", list(TOPIC_CONTEXTS.keys())),
    ("sorry", list(TOPIC_CONTEXTS.keys())),
    ("my bad", list(TOPIC_CONTEXTS.keys())),
    ("oops", list(TOPIC_CONTEXTS.keys())),
    ("wait what", list(TOPIC_CONTEXTS.keys())),
    ("hold on", list(TOPIC_CONTEXTS.keys())),
    ("one sec", list(TOPIC_CONTEXTS.keys())),
]

# Non-continuity messages — topic changes that SHOULD trigger boundaries
# Each tuple: (message_text, incompatible_context_keys)
NON_CONTINUITY_CHANGES = [
    # Technical from personal
    ("Can you explain how quantum computing works?", ["getting_to_know", "personal_history", "relationships", "pets_and_animals", "food_discussion", "travel_stories", "weather_and_seasons"]),
    ("What's the difference between TCP and UDP?", ["getting_to_know", "personal_history", "relationships", "pets_and_animals", "food_discussion", "travel_stories", "weather_and_seasons", "music_chat"]),
    ("How does blockchain technology actually work?", ["getting_to_know", "personal_history", "relationships", "pets_and_animals", "food_discussion", "travel_stories", "weather_and_seasons", "fitness_talk"]),
    ("What programming language should I learn first?", ["personal_history", "relationships", "pets_and_animals", "food_discussion", "travel_stories", "weather_and_seasons", "fitness_talk", "music_chat"]),

    # Food from technical
    ("What's the best recipe for chocolate cake?", ["tech_discussion", "work_chat", "fitness_talk", "education_learning"]),
    ("I'm craving sushi right now", ["tech_discussion", "work_chat", "education_learning"]),
    ("Where can I find good Mexican food around here?", ["tech_discussion", "work_chat", "education_learning", "music_chat"]),

    # Emergency / crisis from casual
    ("My house is on fire what do I do", ["food_discussion", "music_chat", "movies_and_tv", "weather_and_seasons", "pets_and_animals", "travel_stories"]),
    ("I think I'm having a medical emergency", ["food_discussion", "music_chat", "movies_and_tv", "weather_and_seasons", "tech_discussion", "travel_stories"]),
    ("Someone just broke into my car", ["food_discussion", "music_chat", "movies_and_tv", "weather_and_seasons", "tech_discussion", "education_learning"]),

    # Entertainment from work
    ("Did you see the latest Marvel movie?", ["work_chat", "tech_discussion", "fitness_talk", "health_wellness", "education_learning"]),
    ("I need some good book recommendations", ["work_chat", "fitness_talk", "weather_and_seasons"]),
    ("What video games are you playing right now?", ["work_chat", "health_wellness", "education_learning", "food_discussion"]),

    # Travel from fitness
    ("I want to plan a trip to Iceland", ["fitness_talk", "tech_discussion", "work_chat", "education_learning"]),
    ("What airlines do you recommend for Europe?", ["fitness_talk", "tech_discussion", "work_chat", "music_chat"]),

    # Pets from tech
    ("My cat just had kittens!", ["tech_discussion", "work_chat", "education_learning"]),
    ("I'm thinking about adopting a rescue dog", ["tech_discussion", "work_chat", "education_learning", "music_chat"]),

    # Politics/news from personal
    ("What do you think about the upcoming election?", ["pets_and_animals", "fitness_talk", "music_chat", "food_discussion"]),
    ("Did you hear about the earthquake in Turkey?", ["pets_and_animals", "fitness_talk", "music_chat", "tech_discussion"]),

    # Sports from non-sports
    ("Who do you think will win the Super Bowl?", ["tech_discussion", "food_discussion", "music_chat", "education_learning", "health_wellness"]),
    ("I just started watching Formula 1 racing", ["tech_discussion", "food_discussion", "music_chat", "education_learning"]),

    # Finance from casual
    ("How should I invest my savings?", ["pets_and_animals", "music_chat", "movies_and_tv", "weather_and_seasons", "travel_stories"]),
    ("Cryptocurrency prices are wild right now", ["pets_and_animals", "music_chat", "movies_and_tv", "weather_and_seasons", "food_discussion"]),

    # Home improvement from everything else
    ("I need to fix my leaking faucet", ["tech_discussion", "music_chat", "movies_and_tv", "fitness_talk", "travel_stories"]),
    ("What color should I paint my living room?", ["tech_discussion", "work_chat", "fitness_talk", "education_learning"]),

    # Philosophy/deep from casual
    ("What do you think happens after we die?", ["food_discussion", "fitness_talk", "weather_and_seasons", "work_chat", "tech_discussion"]),
    ("Is free will an illusion?", ["food_discussion", "fitness_talk", "weather_and_seasons", "pets_and_animals"]),

    # Random non-sequiturs
    ("Do fish ever get thirsty?", ["work_chat", "tech_discussion", "education_learning", "music_chat", "fitness_talk"]),
    ("Why is the sky blue?", ["work_chat", "relationships", "food_discussion", "music_chat"]),
    ("How many piano tuners are there in Chicago?", ["food_discussion", "pets_and_animals", "fitness_talk", "weather_and_seasons", "travel_stories"]),
]

# fmt: on


@dataclass
class Sample:
    """One labeled boundary detection sample."""
    context_key: str          # which topic context was used
    context_messages: List[str]
    test_message: str
    expected_boundary: bool   # True = should fire, False = should not


def generate_dataset() -> List[Sample]:
    """Generate all context × test_message combinations."""
    samples = []

    # Continuity samples: test_message should NOT fire boundary
    for msg, compatible_keys in CONTINUITY_SHORT_ANSWERS:
        for ctx_key in compatible_keys:
            if ctx_key in TOPIC_CONTEXTS:
                samples.append(Sample(
                    context_key=ctx_key,
                    context_messages=TOPIC_CONTEXTS[ctx_key],
                    test_message=msg,
                    expected_boundary=False,
                ))

    # Non-continuity samples: test_message SHOULD fire boundary
    for msg, incompatible_keys in NON_CONTINUITY_CHANGES:
        for ctx_key in incompatible_keys:
            if ctx_key in TOPIC_CONTEXTS:
                samples.append(Sample(
                    context_key=ctx_key,
                    context_messages=TOPIC_CONTEXTS[ctx_key],
                    test_message=msg,
                    expected_boundary=True,
                ))

    return samples


# ── Embedding cache ──────────────────────────────────────────────────────────

def precompute_embeddings(samples: List[Sample], embed_svc: EmbeddingService):
    """Pre-embed all unique texts. Returns {text: np.ndarray}."""
    unique_texts = set()
    for s in samples:
        for msg in s.context_messages:
            unique_texts.add(msg)
        unique_texts.add(s.test_message)

    print(f"Embedding {len(unique_texts)} unique texts...")
    t0 = time.time()

    cache = {}
    texts = list(unique_texts)
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_svc.generate_embeddings_batch(batch)
        for text, emb in zip(batch, embeddings):
            cache[text] = emb

    elapsed = time.time() - t0
    print(f"Embedded {len(unique_texts)} texts in {elapsed:.1f}s "
          f"({len(unique_texts)/elapsed:.0f} texts/sec)")
    return cache


# ── Evaluation ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Metrics for one parameter configuration."""
    threshold_k: float
    window_size: int
    tp: int  # true positive (boundary detected, was boundary)
    fp: int  # false positive (boundary detected, was NOT boundary) — THE BUG
    tn: int  # true negative (no boundary, was NOT boundary)
    fn: int  # false negative (no boundary, was boundary)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False positive rate — what fraction of continuity messages fire boundaries."""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0


def evaluate_params(
    samples: List[Sample],
    emb_cache: dict,
    threshold_k: float,
    window_size: int,
    short_gate: int = 3,
) -> EvalResult:
    """Run all samples through the boundary detector with given params."""
    # Mock the MemoryStore for clean state per sample
    from services.memory_store import MemoryStore
    from unittest.mock import patch

    tp = fp = tn = fn = 0

    # Temporarily override the module-level SHORT_MSG_MAX_WORDS
    old_gate = _boundary_mod.SHORT_MSG_MAX_WORDS
    _boundary_mod.SHORT_MSG_MAX_WORDS = short_gate

    with patch('services.memory_store.get_shared_store', return_value=MemoryStore()):
        for sample in samples:
            # Fresh detector per sample
            svc = TwoSignalBoundaryService(
                thread_id=f'cal-{id(sample)}',
                window_size=window_size,
                threshold_k=threshold_k,
            )

            # Feed context messages
            for msg in sample.context_messages:
                emb = emb_cache[msg]
                svc.update(emb, msg)

            # Evaluate test message
            test_emb = emb_cache[sample.test_message]
            result = svc.update(test_emb, sample.test_message)

            if sample.expected_boundary:
                if result.is_boundary:
                    tp += 1
                else:
                    fn += 1
            else:
                if result.is_boundary:
                    fp += 1
                else:
                    tn += 1

    # Restore module-level gate
    _boundary_mod.SHORT_MSG_MAX_WORDS = old_gate

    return EvalResult(
        threshold_k=threshold_k,
        window_size=window_size,
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Boundary Detector Calibration")
    print("=" * 70)

    # Generate dataset
    samples = generate_dataset()
    n_continuity = sum(1 for s in samples if not s.expected_boundary)
    n_boundary = sum(1 for s in samples if s.expected_boundary)
    print(f"\nDataset: {len(samples)} samples "
          f"({n_continuity} continuity, {n_boundary} boundary)")

    # Pre-compute embeddings
    embed_svc = EmbeddingService()
    emb_cache = precompute_embeddings(samples, embed_svc)

    # Parameter grid
    k_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    window_values = [4, 5, 6, 7]
    gate_values = [0, 2, 3, 4, 5]  # 0 = no gate

    total = len(k_values) * len(window_values) * len(gate_values)
    print(f"\nSweeping {len(k_values)} K x {len(window_values)} win x "
          f"{len(gate_values)} gate = {total} combinations...\n")

    # Header
    print(f"{'K':>5}  {'Win':>3}  {'Gate':>4}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
          f"{'FPR':>6}  {'Acc':>6}  {'TP':>4}  {'FP':>4}  {'TN':>5}  {'FN':>4}")
    print("-" * 82)

    results = []
    best_f1 = 0.0
    best_result = None
    best_gate = 0

    for k in k_values:
        for w in window_values:
            for g in gate_values:
                result = evaluate_params(samples, emb_cache, k, w, short_gate=g)
                results.append((result, g))

                # Track best by F1 with FPR tiebreaker
                if result.f1 > best_f1 or (result.f1 == best_f1 and
                                            best_result and result.fpr < best_result.fpr):
                    best_f1 = result.f1
                    best_result = result
                    best_gate = g

                marker = " ◀" if result is best_result else ""
                print(f"{k:5.1f}  {w:3d}  {g:4d}  {result.precision:6.3f}  {result.recall:6.3f}  "
                      f"{result.f1:6.3f}  {result.fpr:6.3f}  {result.accuracy:6.3f}  "
                      f"{result.tp:4d}  {result.fp:4d}  {result.tn:5d}  {result.fn:4d}"
                      f"{marker}")

    # Summary
    print("\n" + "=" * 70)
    print("  OPTIMAL PARAMETERS")
    print("=" * 70)
    if best_result:
        print(f"  threshold_k = {best_result.threshold_k}")
        print(f"  window_size = {best_result.window_size}")
        print(f"  short_msg_gate = {best_gate} words")
        print(f"  F1 = {best_result.f1:.4f}")
        print(f"  Precision = {best_result.precision:.4f} "
              f"(boundary detected when it should)")
        print(f"  Recall = {best_result.recall:.4f} "
              f"(boundary caught vs missed)")
        print(f"  FPR = {best_result.fpr:.4f} "
              f"(false boundaries on continuity — THE BUG)")
        print(f"  Accuracy = {best_result.accuracy:.4f}")
        print(f"  TP={best_result.tp}  FP={best_result.fp}  "
              f"TN={best_result.tn}  FN={best_result.fn}")

    # Current vs optimal
    print(f"\n  Current settings: threshold_k=1.6, window_size=5, gate=3")
    current = next((r for r, g in results
                     if r.threshold_k == 1.6 and r.window_size == 5 and g == 3), None)
    if current:
        print(f"  Current F1={current.f1:.4f}  FPR={current.fpr:.4f}  "
              f"Acc={current.accuracy:.4f}")
        if best_result and best_result is not current:
            delta_fpr = current.fpr - best_result.fpr
            delta_f1 = best_result.f1 - current.f1
            print(f"  Improvement: F1 +{delta_f1:.4f}, FPR -{delta_fpr:.4f}")

    # Also show the best at each gate level
    print(f"\n  Best F1 per gate level:")
    for g in gate_values:
        gate_results = [(r, g2) for r, g2 in results if g2 == g]
        if gate_results:
            best_for_gate = max(gate_results, key=lambda x: x[0].f1)
            r = best_for_gate[0]
            print(f"    gate={g}: k={r.threshold_k} w={r.window_size} "
                  f"F1={r.f1:.4f} FPR={r.fpr:.4f} Acc={r.accuracy:.4f}")

    # Print false positives/negatives at OPTIMAL settings
    opt_k = best_result.threshold_k if best_result else 1.6
    opt_w = best_result.window_size if best_result else 5
    opt_g = best_gate

    print(f"\n{'=' * 70}")
    print(f"  FALSE POSITIVE ANALYSIS (optimal: k={opt_k}, w={opt_w}, gate={opt_g})")
    print("=" * 70)

    from services.memory_store import MemoryStore
    from unittest.mock import patch

    _boundary_mod.SHORT_MSG_MAX_WORDS = opt_g
    fp_samples = []
    with patch('services.memory_store.get_shared_store', return_value=MemoryStore()):
        for sample in samples:
            if sample.expected_boundary:
                continue
            svc = TwoSignalBoundaryService(
                thread_id=f'fp-{id(sample)}',
                window_size=opt_w,
                threshold_k=opt_k,
            )
            for msg in sample.context_messages:
                svc.update(emb_cache[msg], msg)

            result = svc.update(emb_cache[sample.test_message], sample.test_message)
            if result.is_boundary:
                fp_samples.append((sample, result))

    if fp_samples:
        print(f"\n  {len(fp_samples)} false positives found:\n")
        for sample, result in fp_samples[:20]:
            wc = len(sample.test_message.split())
            print(f"  [{wc}w] Context: {sample.context_key}")
            print(f"  Message: \"{sample.test_message}\"")
            print(f"  Trigger: {result.trigger}  "
                  f"consec={result.consec_sim:.4f}  "
                  f"window={result.window_sim:.4f}  "
                  f"marker={result.marker_found}")
            print()
    else:
        print("\n  No false positives! Zero FPR achieved.")

    print(f"\n{'=' * 70}")
    print(f"  FALSE NEGATIVE ANALYSIS (optimal: k={opt_k}, w={opt_w}, gate={opt_g})")
    print("=" * 70)

    fn_samples = []
    with patch('services.memory_store.get_shared_store', return_value=MemoryStore()):
        for sample in samples:
            if not sample.expected_boundary:
                continue
            svc = TwoSignalBoundaryService(
                thread_id=f'fn-{id(sample)}',
                window_size=opt_w,
                threshold_k=opt_k,
            )
            for msg in sample.context_messages:
                svc.update(emb_cache[msg], msg)

            result = svc.update(emb_cache[sample.test_message], sample.test_message)
            if not result.is_boundary:
                fn_samples.append((sample, result))

    if fn_samples:
        print(f"\n  {len(fn_samples)} false negatives (missed boundaries):\n")
        for sample, result in fn_samples[:20]:
            print(f"  Context: {sample.context_key}")
            print(f"  Message: \"{sample.test_message}\"")
            print(f"  consec={result.consec_sim:.4f}  "
                  f"window={result.window_sim:.4f}")
            print()
    else:
        print("\n  No false negatives! All topic changes detected.")

    _boundary_mod.SHORT_MSG_MAX_WORDS = 3  # restore default


if __name__ == '__main__':
    main()
