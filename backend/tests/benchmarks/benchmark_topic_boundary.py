"""
Topic Boundary Detection Benchmark — 325-message reliability test.

Feeds a realistic 325-message conversation through multiple detectors
using real embeddings, measuring precision/recall/F1 for boundary detection.

Scenarios covered:
  - Stable topic continuation (false positive resistance)
  - Clear topic switches (true positive detection)
  - Gradual semantic drift (borderline — flexible scoring)
  - Brief asides and topic re-entry
  - Rapid topic switching (stress test)
  - Related topics (discrimination)
  - Discourse marker FP traps (markers within continuations)
  - Very short messages (texting style)
  - Extended stable segments (15+ messages)
  - No-marker topic switches (embedding-only detection)

Requires: gte-modernbert-base ONNX model.
Run:      pytest backend/tests/benchmarks/benchmark_topic_boundary.py -v -s
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ground truth labels
# ─────────────────────────────────────────────────────────────────────────────
CONTINUATION = 'continuation'
BOUNDARY = 'boundary'
BORDERLINE = 'borderline'

# Tolerance: detector may fire up to N messages after a true boundary
TOLERANCE_WINDOW = 2

# Minimum acceptable scores (assert thresholds)
MIN_PRECISION = 0.55
MIN_RECALL = 0.50
MIN_F1 = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Conversation data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    topic: str
    messages: List[str]
    label_overrides: Dict[int, str] = field(default_factory=dict)


SEGMENTS: List[Segment] = [
    # 0: Morning plans (8 msgs — first topic, no boundary)
    Segment('morning_plans', [
        "Good morning! Just waking up and thinking about my day ahead",
        "I need coffee first, then maybe eggs and toast for breakfast",
        "Should I go for a run this morning or save it for the evening?",
        "Let me check my calendar to plan around my meetings today",
        "I have a standup at 9 and a design review at 2 this afternoon",
        "That leaves the morning pretty open for some focused deep work",
        "I should block out 10 to noon for the API migration task",
        "I also need to reply to those emails I got yesterday evening",
    ]),

    # 1: Work/API project (10 msgs, clear boundary)
    Segment('work_api_project', [
        "Actually let me tell you about this API migration I'm working on",
        "The OAuth2 flow needs to handle both mobile and web clients properly",
        "We've got a race condition in the token refresh logic right now",
        "When two requests hit the refresh endpoint simultaneously both get new tokens",
        "But only one gets persisted so the other client gets logged out unexpectedly",
        "I think we need either a mutex or some kind of token versioning scheme",
        "Redis could work as a distributed lock for the refresh operation",
        "Or we could use database-level optimistic locking with version numbers",
        "Let me sketch out the sequence diagram for both approaches",
        "The Redis approach is simpler but it adds an infrastructure dependency",
    ]),

    # 2: Cooking dinner (8 msgs, clear boundary)
    Segment('cooking_dinner', [
        "Completely different topic — what should I cook for dinner tonight?",
        "I've got chicken, rice, and some vegetables in the fridge already",
        "A stir fry would be quick and easy on a weeknight",
        "I need to pick up some soy sauce and sesame oil from the store",
        "Or I could make a chicken curry with coconut milk instead",
        "The curry takes longer but makes amazing leftovers for lunch",
        "Let me check if I have enough garam masala and turmeric",
        "The curry wins — I'll start the rice cooker at 6 and prep at 5:30",
    ]),

    # 3: Weather aside (3 msgs, clear boundary — brief aside)
    Segment('weather_aside', [
        "Oh wait, I just looked outside and it's really dark and cloudy",
        "The forecast says thunderstorms all afternoon with high winds",
        "I should probably skip the grocery run and use what I have at home",
    ]),

    # 4: Return to cooking (5 msgs, boundary — topic re-entry)
    Segment('cooking_return', [
        "OK so back to dinner — I'll make do with what's in the pantry",
        "I found some canned tomatoes and dried lentils which is perfect",
        "A red lentil dal could work really well with the rice I have",
        "I just need to sauté some onions and garlic as a base",
        "This is actually better than the curry — faster and healthier too",
    ]),

    # 5: Gradual drift — cooking → nutrition → fitness (8 msgs, borderline)
    Segment('gradual_nutrition_fitness', [
        "The dal is packed with protein and fiber which is great for my diet",
        "I've been trying to eat more plant-based protein sources lately",
        "My nutritionist said I should aim for about 30 grams of protein per meal",
        "Tracking macros has been really eye-opening for understanding what I eat",
        "I downloaded this app that scans barcodes and logs everything automatically",
        "My trainer says protein intake matters more than total calories for building muscle",
        "Speaking of training, I really need to get back to my gym routine",
        "I've been inconsistent the past two weeks and I can feel the difference",
    ], label_overrides={0: BORDERLINE, 1: BORDERLINE, 6: BORDERLINE, 7: BORDERLINE}),

    # 6: Travel planning (10 msgs, clear boundary)
    Segment('travel_planning', [
        "Hey, totally unrelated — have you helped me plan any trips before?",
        "I'm thinking about visiting Japan next spring during cherry blossom season",
        "The flights from here are expensive though, usually around 1200 round trip",
        "I could save a lot by flying into Osaka instead of Tokyo",
        "Then take the Shinkansen bullet train between cities",
        "I want to spend at least three days in Kyoto visiting the temples",
        "The bamboo grove at Arashiyama is supposed to be incredible in person",
        "I should start looking at hotels now before peak season prices kick in",
        "A ryokan traditional inn would be an amazing experience for a few nights",
        "The food alone is worth the trip — ramen, sushi, wagyu, yakitori",
    ]),

    # 7: Language learning (6 msgs, borderline boundary — related to travel)
    Segment('language_learning', [
        "I should really learn some basic Japanese before the trip",
        "I've been using Duolingo on and off but I'm not making much progress",
        "The writing system is the hardest part — three different scripts",
        "Hiragana and katakana are manageable but kanji is a whole other level",
        "Maybe I should focus on just conversational phrases for now",
        "Even just knowing how to order food and ask for directions would help",
    ], label_overrides={0: BORDERLINE}),

    # 8: Programming/tech (8 msgs, clear boundary)
    Segment('programming', [
        "Let me switch gears — I've been learning Rust in my spare time",
        "The borrow checker is brutal but it catches so many bugs at compile time",
        "I rewrote a Python script in Rust and it runs 50 times faster now",
        "The error handling with Result types is so much cleaner than try-except",
        "Pattern matching in Rust feels like a superpower once you get used to it",
        "I'm building a CLI tool that processes large log files in parallel",
        "Using rayon for data parallelism made the code almost trivially parallel",
        "The ecosystem is great too — serde for JSON, tokio for async, clap for CLI",
    ]),

    # 9: Music (5 msgs, clear boundary)
    Segment('music', [
        "What kind of music do you think would be good for coding focus?",
        "I usually listen to lo-fi hip hop or ambient electronic while working",
        "Recently discovered this artist called Tycho and their sound is perfect",
        "Something about instrumental music without lyrics helps me concentrate",
        "I made a whole Spotify playlist just for deep work sessions",
    ]),

    # 10: Movies (5 msgs, clear boundary)
    Segment('movies', [
        "Did you catch any good movies recently? I watched Dune Part Two last week",
        "The cinematography was absolutely stunning in IMAX format",
        "Denis Villeneuve really knows how to create scale and atmosphere",
        "I want to rewatch Blade Runner 2049 now since it's the same director",
        "The whole sci-fi genre has been having a renaissance lately",
    ]),

    # 11: Books (5 msgs, clear boundary)
    Segment('books', [
        "Speaking of sci-fi I've been reading the Three-Body Problem trilogy",
        "Liu Cixin's imagination is on another level with the physics concepts",
        "The Dark Forest theory is genuinely unsettling as a Fermi paradox solution",
        "I read about 30 pages every night before bed as part of my routine",
        "Want to start Piranesi by Susanna Clarke next, heard amazing things",
    ]),

    # 12: Philosophy — extended stable (10 msgs, no explicit boundary)
    Segment('philosophy', [
        "That Dark Forest idea got me thinking about philosophy of mind",
        "Do you think consciousness requires self-awareness or just information processing?",
        "The hard problem of consciousness is fascinating — why does experience feel like something?",
        "David Chalmers makes a compelling case that you can't reduce qualia to physics",
        "But Daniel Dennett argues consciousness is basically a bag of tricks",
        "I lean toward integrated information theory — phi as a measure of consciousness",
        "It would mean even simple systems have some tiny amount of experience",
        "Which connects to panpsychism, the idea that mind is fundamental to matter",
        "Thomas Nagel's What Is It Like to Be a Bat paper is still the best starting point",
        "The debate might be an artifact of how language fails to describe subjective experience",
    ]),

    # 13: Relationships (8 msgs, clear boundary)
    Segment('relationships', [
        "On a personal note, my partner and I have been talking about moving in together",
        "We've been dating about two years and it feels like the right time",
        "The tricky part is we both have apartments in different neighborhoods",
        "Whose lease do we break? Or do we find a completely new place?",
        "A new place feels more equitable — ours, not yours that I moved into",
        "We're looking at two-bedroom apartments so we each have a home office",
        "The budget is tight though — rent has gone up like 20 percent this year",
        "We'll start apartment hunting next month when some leases open up",
    ]),

    # 14: Home renovation (8 msgs, clear boundary)
    Segment('home_renovation', [
        "That reminds me, my parents just finished renovating their kitchen",
        "They ripped out old cabinets and put in beautiful white oak ones",
        "The countertops are quartz which is basically indestructible",
        "They added a kitchen island with seating for four that opens the space",
        "The whole project took about six weeks of eating out constantly",
        "Contractor issues delayed it twice — material shortage then scheduling",
        "Total cost came in about 15 percent over the initial estimate",
        "But the result is gorgeous and probably added value to the house",
    ]),

    # 15: Finances (7 msgs, clear boundary)
    Segment('finances', [
        "All this talk about costs reminds me I need to review my budget",
        "I've been spending way too much on food delivery apps lately",
        "Like 400 dollars a month on DoorDash and UberEats is embarrassing",
        "If I meal prep on Sundays I could probably cut that to 100",
        "I should also look into refinancing my student loans while rates hold",
        "There's a new high-yield savings account offering 5.2 percent APY",
        "I want to hit 10000 in my emergency fund by the end of this year",
    ]),

    # 16: Sports aside (2 msgs, clear boundary — very brief)
    Segment('sports_aside', [
        "Oh my god, did you see the game last night? That buzzer-beater was insane!",
        "Three-pointer from half court to win by one in overtime, absolutely unreal",
    ]),

    # 17: Return to finances (5 msgs, boundary — topic re-entry)
    Segment('finances_return', [
        "Anyway back to the budget — I set up automatic transfers to savings",
        "Every paycheck, 20 percent goes straight to savings before I touch it",
        "Pay yourself first is the simplest financial advice that actually works",
        "I'm also maxing out my 401k match because that's literally free money",
        "By end of year I should be in a much better position financially",
    ]),

    # 18: Pet care (7 msgs, clear boundary)
    Segment('pet_care', [
        "My cat has been acting weird lately, I'm a little worried about her",
        "She's been hiding under the bed more than usual and barely eating",
        "She's normally super social and always wants attention and treats",
        "I'm wondering if it's stress from the construction noise next door",
        "Or maybe she ate something she shouldn't have from the house plants",
        "I moved all the toxic plants out of reach last month but she's crafty",
        "I think I should take her to the vet this week just to be safe",
    ]),

    # 19: Gradual drift — pet → vet → health insurance (5 msgs, borderline)
    Segment('gradual_vet_health', [
        "The vet visits are getting really expensive without pet insurance",
        "I looked into pet insurance plans and they range from 30 to 80 a month",
        "It's kind of like human health insurance — high deductibles, limited coverage",
        "That reminds me I need to review my own health insurance during open enrollment",
        "My current plan has a 3000 dollar deductible which feels really high",
    ], label_overrides={0: BORDERLINE, 1: BORDERLINE, 3: BORDERLINE, 4: BORDERLINE}),

    # 20: Holiday planning (8 msgs, clear boundary)
    Segment('holiday_planning', [
        "Christmas is coming up fast and I haven't started shopping at all",
        "I usually make a list in November but this year I've been so busy",
        "My family does Secret Santa so I only need one big gift",
        "I drew my sister's name and she's impossible to shop for honestly",
        "She said she wants experiences not things which is very helpful",
        "Maybe I'll get her tickets to a concert or a cooking class",
        "I also need to figure out who's hosting the big dinner this year",
        "Last year we did it at my parents' newly renovated kitchen",
    ]),

    # 21: Gardening (7 msgs, clear boundary)
    Segment('gardening', [
        "I started a small herb garden on my balcony and it's going well",
        "Basil, rosemary, thyme, and mint all love the afternoon sun",
        "The basil is growing so fast I can't use it all, should make pesto",
        "I tried growing tomatoes too but they need more direct sunlight",
        "Container gardening is limited by space but it's so satisfying",
        "I water everything in the morning and check soil moisture daily",
        "Next spring I want to try growing peppers and maybe strawberries",
    ]),

    # 22: Career goals (7 msgs, clear boundary)
    Segment('career_goals', [
        "I've been thinking a lot about where I want to be in five years",
        "I'm a senior engineer but I'm not sure I want to go into management",
        "The IC track seems more appealing — staff engineer, principal",
        "But the impact you can have as a manager on team culture is meaningful",
        "My mentor suggested I try a tech lead role first as a middle ground",
        "That way I get leadership experience without fully leaving code",
        "I'll talk to my manager about it at our next one-on-one",
    ]),

    # 23: Gaming (5 msgs, clear boundary)
    Segment('gaming', [
        "I need a good game recommendation, been so bored in the evenings",
        "I finished Baldur's Gate 3 and now everything else feels bland",
        "Thinking about trying Elden Ring but I hear it's brutally difficult",
        "Maybe something more relaxing like Stardew Valley instead",
        "There's also the new Zelda which everyone says is a masterpiece",
    ]),

    # 24: Streaming/TV (5 msgs, clear boundary)
    Segment('streaming_tv', [
        "Or maybe I should just watch more TV instead of gaming",
        "I'm three seasons behind on Succession and people keep spoiling it",
        "The new season of True Detective is supposed to be a return to form",
        "I've been hearing great things about Shogun on Hulu",
        "There's too much good content now and not enough time to watch it",
    ]),

    # 25: Sleep and wellness (8 msgs, clear boundary)
    Segment('sleep_wellness', [
        "Part of my problem is I stay up way too late watching shows and gaming",
        "My sleep schedule is completely wrecked, going to bed at 1am most nights",
        "I read that consistent sleep timing matters more than total hours",
        "I bought a sunrise alarm clock that gradually brightens the room",
        "It's supposed to make waking up more natural than a jarring alarm",
        "I'm also trying to cut screen time an hour before bed but it's hard",
        "Blue light glasses help a bit but not scrolling would help more",
        "My goal is lights out by 11pm and wake up naturally by 7am",
    ]),

    # 26: Childhood memories (9 msgs, clear boundary)
    Segment('childhood_memories', [
        "All this self-improvement stuff makes me nostalgic for being a kid",
        "When I was little I used to build elaborate blanket forts in the living room",
        "My brother and I would pretend the floor was lava and jump between cushions",
        "We had this one summer where we built a treehouse in the backyard",
        "Dad helped with the platform but we did the walls and roof ourselves",
        "It survived about two winters before a big storm knocked it down",
        "Those are the kind of memories that really stick with you decades later",
        "I should call my brother this weekend, we haven't talked in weeks",
        "Maybe we can plan a camping trip this summer like we used to do",
    ]),

    # 27: Current events / news (7 msgs, clear boundary)
    Segment('current_events', [
        "Have you been following the news about the new space telescope data?",
        "They found evidence of phosphine in the atmosphere of a nearby exoplanet",
        "That could potentially indicate biological processes happening there",
        "The scientific community is being cautious but the data looks promising",
        "It reminds me of the Venus phosphine debate from a few years back",
        "Space exploration funding has been increasing which is encouraging",
        "I think we'll see a crewed Mars mission within the next fifteen years",
    ]),

    # 28: Photography hobby (6 msgs, clear boundary)
    Segment('photography', [
        "I just bought a used mirrorless camera and I'm obsessed with it",
        "The autofocus on these newer Sony bodies is absolutely incredible",
        "I've been shooting mostly street photography on my walks around the city",
        "Golden hour light makes even mundane buildings look cinematic",
        "I need to learn Lightroom properly instead of just using auto presets",
        "There's a local photography club that meets monthly I should join",
    ]),

    # 29: Weekend plans (5 msgs, clear boundary)
    Segment('weekend_plans', [
        "So this weekend I think I'm going to try to be really intentional",
        "Saturday morning I'll do the farmers market and meal prep for the week",
        "Then maybe hit the gym in the afternoon and take the camera out after",
        "Sunday I want to just relax, read, maybe call my brother like I said",
        "It feels good having a loose plan instead of just letting the weekend happen",
    ]),

    # ── NEW SEGMENTS: hard cases ─────────────────────────────────────────────

    # 30: Extended database deep-dive (15 msgs, boundary — stability test)
    #     Contains "By the way" on msg 6 as FP trap (same topic continuation)
    Segment('database_optimization', [
        "I've been deep-diving into database optimization at work this week",
        "Our main query was taking 3 seconds because it was doing a full table scan",
        "Adding a composite index on user_id and created_at brought it down to 12ms",
        "The EXPLAIN ANALYZE output is so useful for understanding query plans",
        "PostgreSQL's query planner is really smart about choosing between index types",
        "B-tree indexes work great for range queries but GIN is better for full-text",
        "By the way, the partial index trick saved us a ton of storage space",
        "We only index rows where status is active which is about 10 percent of the table",
        "Connection pooling with PgBouncer was another huge win for us",
        "We went from 500 direct connections to 20 pooled connections",
        "The latency improvement was dramatic especially under high concurrency",
        "I also set up automated VACUUM and ANALYZE schedules for maintenance",
        "Dead tuple bloat was causing our table sizes to grow out of control",
        "Monitoring pg_stat_user_tables helps catch performance issues early",
        "Next I want to look into table partitioning for our events table",
    ]),

    # 31: No-marker switch to apartment cleaning (11 msgs, boundary, NO marker)
    #     Contains "That reminds me" on msg 5 as FP trap (same topic continuation)
    Segment('apartment_cleaning', [
        "I need to deep clean my apartment before my parents visit next week",
        "The bathroom grout needs scrubbing and the kitchen hood is greasy",
        "I should also steam clean the carpets since the cat sheds everywhere",
        "Marie Kondo says to tackle categories not rooms but that never works for me",
        "Realistically I need two full days to get everything presentable",
        "That reminds me I need to buy more cleaning supplies from the store",
        "The all-purpose cleaner ran out last week and I've been using dish soap",
        "I should get one of those Swiffer mops for the hardwood floors",
        "Microfiber cloths are way better than paper towels for windows",
        "My friend swears by those cleaning concentrate tablets you dissolve",
        "They're better for the environment and way cheaper per bottle",
    ]),

    # 32: Very short texting-style messages (6 msgs, boundary — short msg test)
    Segment('food_ordering', [
        "Ugh I'm starving",
        "Pizza or tacos?",
        "I think tacos win tonight",
        "Extra guac for sure",
        "This place on 5th has amazing al pastor",
        "Ordering now, should be here in 30 minutes",
    ]),

    # 33: Return to Japan travel (5 msgs, boundary — topic return to seg 6)
    Segment('japan_travel_return', [
        "I just found amazing flight deals to Tokyo for next April",
        "Round trip for 850 dollars which is way less than I expected",
        "The cherry blossom forecast says early April is peak bloom in Tokyo",
        "I already bookmarked three ryokans in Hakone with mountain views",
        "This trip is actually happening and I'm getting so excited about it",
    ]),

    # 34: Machine learning (8 msgs, boundary — related but distinct from programming seg 8)
    Segment('machine_learning', [
        "I started taking an online course on machine learning fundamentals",
        "The math behind gradient descent is simpler than I expected",
        "Linear regression is basically just fitting a line by minimizing error",
        "Neural networks are where it gets wild with all the layers and activation functions",
        "I trained a small model to classify handwritten digits and got 97 percent accuracy",
        "The hardest part is getting good training data not writing the model code",
        "Data augmentation techniques like rotation and cropping help a lot with small datasets",
        "I want to try building a recommendation engine for my movie watchlist next",
    ]),

    # 35: No-marker switch to commuting (10 msgs, boundary, NO marker)
    #     Contains "Anyway" on msg 6 as FP trap (same topic continuation)
    Segment('commuting', [
        "The morning commute has been absolutely terrible this month",
        "Construction on the highway is adding 20 minutes each way",
        "I've been taking the side roads through residential streets instead",
        "Working from home two days a week has been a lifesaver honestly",
        "The gas prices make the commute even more painful at 4 dollars a gallon",
        "I calculated I spend about 250 dollars a month just on gas for work",
        "Anyway the hybrid car I've been looking at would cut that in half",
        "The Toyota RAV4 hybrid gets like 40 miles per gallon in the city",
        "My lease is up in three months so the timing works out perfectly",
        "Electric would be even better but the charging infrastructure isn't there yet",
    ]),

    # 36: Quick switch to podcasts (3 msgs, boundary — brief segment)
    Segment('podcasts', [
        "I discovered an amazing podcast about behavioral economics",
        "Each episode breaks down a cognitive bias with real world examples",
        "It makes my drive way more productive instead of just sitting in traffic",
    ]),

    # 37: Quick switch to social media (3 msgs, boundary — rapid switching)
    Segment('social_media', [
        "I've been trying to reduce my social media screen time drastically",
        "I deleted Twitter and Instagram from my phone last week",
        "Already spending that time reading instead which feels way better",
    ]),

    # 38: Extended personal reflection (12 msgs, boundary — stability test)
    Segment('life_reflection', [
        "Sometimes I wonder if I'm living the life I actually want or just defaulting",
        "I have a good job and comfortable routine but is that enough?",
        "My twenties were all about career building and now I'm not sure what's next",
        "I see friends starting families and buying houses and feel behind somehow",
        "But comparison is the thief of joy as they say",
        "What I really value is freedom and creative expression more than stability",
        "Maybe I should take a sabbatical and travel for a few months",
        "The savings I've built up could cover six months easily",
        "But then what? Return to the same routine or start something entirely new?",
        "I think the real question is what would I regret not doing in ten years",
        "Probably not taking the risk while I'm young enough to recover from it",
        "I should journal about this more instead of just thinking in circles",
    ]),

    # 39: No-marker switch to public transport (10 msgs, boundary, NO marker)
    #     Contains "Speaking of" on msg 5 as FP trap (same topic continuation)
    Segment('public_transport', [
        "The new light rail extension finally opened near my neighborhood",
        "It takes about 25 minutes to get downtown which is faster than driving",
        "The monthly pass is only 85 dollars compared to 250 in gas and parking",
        "I love being able to read or listen to podcasts instead of focusing on traffic",
        "More cities need to invest in public transit instead of just widening highways",
        "Speaking of public transit, the bus routes got redesigned last month",
        "They added express routes that skip half the stops during rush hour",
        "The real-time tracking app actually works now unlike the old system",
        "I can see exactly when the next bus arrives which eliminates guessing",
        "The whole network feels more connected and usable than it used to be",
    ]),

    # 40: Return to cooking theme (5 msgs, boundary — topic return to seg 2/4)
    Segment('cooking_return_2', [
        "I finally tried making that red lentil dal recipe I was thinking about",
        "It turned out amazing with just onions garlic ginger and canned tomatoes",
        "The secret is toasting the spices in oil before adding the lentils",
        "I made enough for four portions so I've got lunch sorted all week",
        "Batch cooking on Sunday is the single best habit I've built this year",
    ]),

    # 41: Restaurant reviews — related but distinct from cooking (6 msgs, borderline)
    Segment('restaurant_reviews', [
        "There's a new ramen place downtown that everyone's been raving about",
        "The tonkotsu broth is apparently simmered for 18 hours",
        "I prefer miso ramen personally but good tonkotsu is hard to beat",
        "Their appetizer menu has these incredible pork gyoza too",
        "The prices are reasonable considering the portion sizes",
        "I want to try their lunch special before the hype dies down",
    ], label_overrides={0: BORDERLINE}),

    # 42: No-marker switch to learning piano (10 msgs, boundary, NO marker)
    #     Contains "On a side note" on msg 5 as FP trap (same topic continuation)
    Segment('learning_piano', [
        "I bought a digital piano last month and I've been practicing every day",
        "Starting with simple scales and chord progressions to build muscle memory",
        "YouTube tutorials are surprisingly effective for learning basic songs",
        "I can play the intro to Clair de Lune which is really motivating",
        "My neighbor hasn't complained yet so the headphone jack is saving me",
        "On a side note I found this app that listens to you play and scores accuracy",
        "It highlights which notes I'm rushing and which I'm holding too long",
        "The rhythm exercises are harder than the actual notes for me",
        "My left hand is still way weaker than my right for independent playing",
        "I'm setting a goal to learn one full piece by the end of next month",
    ]),

    # 43: AI regulation discussion (10 msgs, boundary — related to philosophy seg 12)
    Segment('ai_discussion', [
        "Have you been following the latest developments in AI regulation?",
        "The EU AI Act is setting some really strict requirements for high-risk systems",
        "Companies using AI for hiring or credit scoring need to prove fairness now",
        "I think transparency requirements are good but compliance costs are huge",
        "Small startups can't afford the same auditing as big tech companies",
        "The open source AI community is worried about being regulated out of existence",
        "There's a real tension between innovation speed and safety guardrails",
        "China and the US have fundamentally different approaches to regulation",
        "I think the EU model is somewhere in between which makes sense",
        "The pace of change makes any regulation feel outdated before it's even enacted",
    ]),

    # 44: No-marker switch to cooking equipment (5 msgs, boundary, NO marker)
    Segment('cooking_equipment', [
        "I just bought a cast iron skillet and it has changed my cooking completely",
        "The heat retention is incredible for getting a perfect sear on steak",
        "Seasoning it properly took a few attempts but now it's basically nonstick",
        "Everyone says cast iron is hard to maintain but it's really not that bad",
        "My next purchase is going to be a Dutch oven for soups and bread baking",
    ]),

    # 45: Gradual drift from equipment to kitchen memories (6 msgs, borderline)
    Segment('kitchen_thoughts', [
        "A good Dutch oven is the kind of thing you keep for decades",
        "My grandmother had one she used for forty years and it was perfect",
        "Her whole kitchen was full of tools she'd accumulated over a lifetime",
        "It makes me think about how my parents redesigned their kitchen recently",
        "Modern kitchens prioritize aesthetics but sometimes lose functionality",
        "I'd rather have a functional kitchen with good tools than a pretty empty one",
    ], label_overrides={0: BORDERLINE, 3: BORDERLINE, 4: BORDERLINE, 5: BORDERLINE}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Flatten segments into labeled messages
# ─────────────────────────────────────────────────────────────────────────────

def _build_labeled_messages() -> List[dict]:
    """Convert SEGMENTS into a flat list of {text, topic, label, segment_idx, msg_idx}."""
    messages = []
    global_idx = 0

    for seg_idx, seg in enumerate(SEGMENTS):
        for local_idx, text in enumerate(seg.messages):
            # Default label: first msg of non-first segment = BOUNDARY, rest = CONTINUATION
            if local_idx in seg.label_overrides:
                label = seg.label_overrides[local_idx]
            elif seg_idx == 0:
                label = CONTINUATION
            elif local_idx == 0:
                label = BOUNDARY
            else:
                label = CONTINUATION

            messages.append({
                'text': text,
                'topic': seg.topic,
                'label': label,
                'segment_idx': seg_idx,
                'msg_idx': global_idx,
            })
            global_idx += 1

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Topic centroid tracker (simulates TopicClassifierService centroid logic)
# ─────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    """Maintains topic centroids as running averages of their member embeddings."""

    def __init__(self):
        self._centroids: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, int] = {}

    def best_similarity(self, embedding: np.ndarray) -> float:
        """Max cosine similarity to any known centroid (0.0 if none)."""
        if not self._centroids:
            return 0.0
        sims = [
            float(np.dot(embedding, c))
            for c in self._centroids.values()
        ]
        return max(sims)

    def update(self, topic: str, embedding: np.ndarray):
        """Update the running centroid for a topic."""
        if topic not in self._centroids:
            self._centroids[topic] = embedding.copy()
            self._counts[topic] = 1
        else:
            n = self._counts[topic]
            self._centroids[topic] = (n * self._centroids[topic] + embedding) / (n + 1)
            # Re-normalise for dot-product cosine
            norm = np.linalg.norm(self._centroids[topic])
            if norm > 1e-8:
                self._centroids[topic] /= norm
            self._counts[topic] = n + 1

    @property
    def topic_count(self) -> int:
        return len(self._centroids)


# ─────────────────────────────────────────────────────────────────────────────
# Two-Signal Detector (Option C)
# ─────────────────────────────────────────────────────────────────────────────

class TwoSignalDetector:
    """Two-signal boundary detector — fires only when both signals agree.

    Signal 1: Consecutive similarity — cos(current, previous message)
    Signal 2: Window similarity — cos(current, centroid of last N messages)

    Both must drop below their self-calibrating threshold (mean - K*std)
    for a boundary to fire. Two independent signals agreeing eliminates
    most false positives.

    Stats are NOT updated when a boundary fires, so topic switches don't
    contaminate the "normal conversation" baseline.
    """

    def __init__(self, window_size: int = 5, cold_start: int = 5, threshold_k: float = 1.5):
        self.window_size = window_size
        self.cold_start = cold_start
        self.threshold_k = threshold_k
        self._embeddings: List[np.ndarray] = []
        self._consec_history: List[float] = []
        self._window_history: List[float] = []

    def update(self, embedding: np.ndarray) -> Tuple[bool, dict]:
        """Process one message. Returns (is_boundary, diagnostics)."""
        n = len(self._embeddings)

        if n == 0:
            self._embeddings.append(embedding)
            return False, {'cold_start': True, 'msg_count': 1}

        # Signal 1: consecutive similarity
        consec_sim = float(np.dot(embedding, self._embeddings[-1]))

        # Signal 2: window centroid similarity
        window = self._embeddings[-self.window_size:]
        centroid = np.mean(window, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        window_sim = float(np.dot(embedding, centroid))

        self._embeddings.append(embedding)

        # Cold start — gather stats but don't fire
        if len(self._consec_history) < self.cold_start:
            self._consec_history.append(consec_sim)
            self._window_history.append(window_sim)
            return False, {
                'cold_start': True,
                'msg_count': n + 1,
                'consec_sim': round(consec_sim, 4),
                'window_sim': round(window_sim, 4),
            }

        # Self-calibrating thresholds from history
        consec_mean = float(np.mean(self._consec_history))
        consec_std = max(float(np.std(self._consec_history)), 0.01)
        window_mean = float(np.mean(self._window_history))
        window_std = max(float(np.std(self._window_history)), 0.01)

        consec_thresh = consec_mean - self.threshold_k * consec_std
        window_thresh = window_mean - self.threshold_k * window_std

        consec_drop = consec_sim < consec_thresh
        window_drop = window_sim < window_thresh

        is_boundary = consec_drop and window_drop

        # Only update stats on non-boundary messages (keep baseline clean)
        if not is_boundary:
            self._consec_history.append(consec_sim)
            self._window_history.append(window_sim)

        return is_boundary, {
            'consec_sim': round(consec_sim, 4),
            'window_sim': round(window_sim, 4),
            'consec_thresh': round(consec_thresh, 4),
            'window_thresh': round(window_thresh, 4),
            'consec_drop': consec_drop,
            'window_drop': window_drop,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    total_messages: int
    total_boundaries: int
    total_continuations: int
    total_borderline: int

    # Exact match
    exact_tp: int
    exact_fp: int
    exact_fn: int

    # Tolerant match (±TOLERANCE_WINDOW msgs)
    tolerant_tp: int
    tolerant_fp: int
    tolerant_fn: int

    # Per-segment detail
    segment_results: List[dict]

    # Timing
    embed_time_s: float
    detect_time_s: float

    @property
    def exact_precision(self) -> float:
        denom = self.exact_tp + self.exact_fp
        return self.exact_tp / denom if denom else 0.0

    @property
    def exact_recall(self) -> float:
        denom = self.exact_tp + self.exact_fn
        return self.exact_tp / denom if denom else 0.0

    @property
    def exact_f1(self) -> float:
        p, r = self.exact_precision, self.exact_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def tolerant_precision(self) -> float:
        denom = self.tolerant_tp + self.tolerant_fp
        return self.tolerant_tp / denom if denom else 0.0

    @property
    def tolerant_recall(self) -> float:
        denom = self.tolerant_tp + self.tolerant_fn
        return self.tolerant_tp / denom if denom else 0.0

    @property
    def tolerant_f1(self) -> float:
        p, r = self.tolerant_precision, self.tolerant_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _score(
    messages: List[dict],
    predictions: List[bool],
) -> Tuple[dict, dict]:
    """Compute exact and tolerant P/R/F1.

    Returns (exact_counts, tolerant_counts) each as {tp, fp, fn}.
    """
    n = len(messages)

    # Identify boundary positions (ground truth)
    boundary_idxs = [i for i, m in enumerate(messages) if m['label'] == BOUNDARY]
    borderline_idxs = set(i for i, m in enumerate(messages) if m['label'] == BORDERLINE)

    # ── Exact scoring ────────────────────────────────────────────────
    exact_tp = 0
    exact_fn = 0
    exact_matched_fires = set()

    for bi in boundary_idxs:
        if predictions[bi]:
            exact_tp += 1
            exact_matched_fires.add(bi)
        else:
            exact_fn += 1

    exact_fp = sum(
        1 for i in range(n)
        if predictions[i]
        and i not in exact_matched_fires
        and i not in borderline_idxs
        and messages[i]['label'] != BOUNDARY
    )

    # ── Tolerant scoring ─────────────────────────────────────────────
    tolerant_tp = 0
    tolerant_fn = 0
    tolerant_matched_fires = set()

    for bi in boundary_idxs:
        found = False
        for offset in range(TOLERANCE_WINDOW + 1):
            check_idx = bi + offset
            if check_idx < n and predictions[check_idx] and check_idx not in tolerant_matched_fires:
                tolerant_tp += 1
                tolerant_matched_fires.add(check_idx)
                found = True
                break
        if not found:
            tolerant_fn += 1

    tolerant_fp = sum(
        1 for i in range(n)
        if predictions[i]
        and i not in tolerant_matched_fires
        and i not in borderline_idxs
        and messages[i]['label'] != BOUNDARY
    )

    return (
        {'tp': exact_tp, 'fp': exact_fp, 'fn': exact_fn},
        {'tp': tolerant_tp, 'fp': tolerant_fp, 'fn': tolerant_fn},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(result: BenchmarkResult):
    """Print a detailed human-readable benchmark report."""
    print("\n" + "=" * 74)
    print("  TOPIC BOUNDARY DETECTION BENCHMARK")
    print("=" * 74)

    print(f"\n  Messages: {result.total_messages}")
    print(f"  Boundaries (ground truth): {result.total_boundaries}")
    print(f"  Continuations: {result.total_continuations}")
    print(f"  Borderline: {result.total_borderline}")
    print(f"  Embedding time: {result.embed_time_s:.2f}s")
    print(f"  Detection time: {result.detect_time_s:.4f}s")

    print(f"\n  {'─' * 40}")
    print(f"  EXACT MATCH")
    print(f"  {'─' * 40}")
    print(f"  TP: {result.exact_tp}  FP: {result.exact_fp}  FN: {result.exact_fn}")
    print(f"  Precision: {result.exact_precision:.3f}")
    print(f"  Recall:    {result.exact_recall:.3f}")
    print(f"  F1:        {result.exact_f1:.3f}")

    print(f"\n  {'─' * 40}")
    print(f"  TOLERANT MATCH (window={TOLERANCE_WINDOW})")
    print(f"  {'─' * 40}")
    print(f"  TP: {result.tolerant_tp}  FP: {result.tolerant_fp}  FN: {result.tolerant_fn}")
    print(f"  Precision: {result.tolerant_precision:.3f}")
    print(f"  Recall:    {result.tolerant_recall:.3f}")
    print(f"  F1:        {result.tolerant_f1:.3f}")

    print(f"\n  {'─' * 40}")
    print(f"  PER-SEGMENT DETAIL")
    print(f"  {'─' * 40}")
    print(f"  {'Segment':<30} {'Expect':>6} {'Fired':>6} {'Result':>8}")
    print(f"  {'─' * 54}")

    for sr in result.segment_results:
        expect = sr['expected_boundary']
        fired = sr['any_fire_in_window']
        if expect == 'boundary':
            verdict = 'TP' if fired else 'MISS'
        elif expect == 'borderline':
            verdict = 'ok' if fired else 'ok'
        else:
            verdict = 'FP!' if fired else 'ok'
        print(f"  {sr['topic']:<30} {expect:>6} {str(fired):>6} {verdict:>8}")

    print("\n" + "=" * 74)

    # Message-level detail for misses and false positives
    misses = [sr for sr in result.segment_results
              if sr['expected_boundary'] == 'boundary' and not sr['any_fire_in_window']]
    fps = [sr for sr in result.segment_results
           if sr['expected_boundary'] == 'none' and sr.get('false_fires', 0) > 0]

    if misses:
        print("\n  MISSED BOUNDARIES:")
        for sr in misses:
            print(f"    [{sr['segment_idx']}] {sr['topic']} — first msg: \"{sr['first_msg'][:60]}...\"")

    if fps:
        print("\n  FALSE POSITIVE SEGMENTS:")
        for sr in fps:
            print(f"    [{sr['segment_idx']}] {sr['topic']} — {sr['false_fires']} false fire(s)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Generic benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_segment_results(messages, predictions, diagnostics):
    """Build per-segment results from flat predictions list."""
    segment_results = []
    msg_offset = 0
    for seg_idx, seg in enumerate(SEGMENTS):
        seg_len = len(seg.messages)
        seg_preds = predictions[msg_offset:msg_offset + seg_len]
        seg_msgs = messages[msg_offset:msg_offset + seg_len]

        if seg_idx == 0:
            expected = 'none'
        elif seg_msgs[0]['label'] == BORDERLINE:
            expected = 'borderline'
        else:
            expected = 'boundary'

        window_end = min(seg_len, 1 + TOLERANCE_WINDOW)
        any_fire = any(seg_preds[:window_end])

        false_fires = 0
        for j in range(seg_len):
            if seg_preds[j] and seg_msgs[j]['label'] == CONTINUATION:
                if j > TOLERANCE_WINDOW or seg_msgs[0]['label'] != BOUNDARY:
                    false_fires += 1

        segment_results.append({
            'segment_idx': seg_idx,
            'topic': seg.topic,
            'expected_boundary': expected,
            'any_fire_in_window': any_fire,
            'false_fires': false_fires,
            'first_msg': seg.messages[0],
            'diagnostics': diagnostics[msg_offset:msg_offset + seg_len],
        })
        msg_offset += seg_len

    return segment_results


def _make_result(messages, predictions, diagnostics, embed_time, detect_time) -> BenchmarkResult:
    """Build BenchmarkResult from predictions."""
    exact, tolerant = _score(messages, predictions)
    segment_results = _build_segment_results(messages, predictions, diagnostics)
    labels = [m['label'] for m in messages]

    return BenchmarkResult(
        total_messages=len(messages),
        total_boundaries=labels.count(BOUNDARY),
        total_continuations=labels.count(CONTINUATION),
        total_borderline=labels.count(BORDERLINE),
        exact_tp=exact['tp'],
        exact_fp=exact['fp'],
        exact_fn=exact['fn'],
        tolerant_tp=tolerant['tp'],
        tolerant_fp=tolerant['fp'],
        tolerant_fn=tolerant['fn'],
        segment_results=segment_results,
        embed_time_s=embed_time,
        detect_time_s=detect_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detector runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_adaptive(messages, embeddings) -> BenchmarkResult:
    """Run the current AdaptiveBoundaryDetector (3-layer accumulator)."""
    from services.adaptive_boundary_detector import AdaptiveBoundaryDetector

    tracker = CentroidTracker()

    mock_store = MagicMock()
    mock_store.get.return_value = None
    with patch('services.adaptive_boundary_detector.get_shared_store', return_value=mock_store):
        detector = AdaptiveBoundaryDetector(thread_id="benchmark")
        predictions = []
        diagnostics = []

        t0 = time.time()
        for i, (m, emb) in enumerate(zip(messages, embeddings)):
            best_sim = tracker.best_similarity(emb)
            result = detector.update(emb, best_sim)
            predictions.append(result.is_boundary)
            diagnostics.append({
                'msg_idx': i,
                'best_sim': round(best_sim, 4),
                'is_boundary': result.is_boundary,
                'acc': round(result.accumulator, 3),
                'bound': round(result.boundary, 3),
            })
            tracker.update(m['topic'], emb)
        detect_time = time.time() - t0

    return _make_result(messages, predictions, diagnostics, 0.0, detect_time)


def _run_static(messages, embeddings, threshold: float = 0.65) -> BenchmarkResult:
    """Run the static threshold detector (best_sim < threshold)."""
    tracker = CentroidTracker()
    predictions = []
    diagnostics = []

    t0 = time.time()
    for i, (m, emb) in enumerate(zip(messages, embeddings)):
        best_sim = tracker.best_similarity(emb)
        is_boundary = best_sim < threshold if tracker.topic_count > 0 else False
        predictions.append(is_boundary)
        diagnostics.append({
            'msg_idx': i,
            'best_sim': round(best_sim, 4),
            'is_boundary': is_boundary,
        })
        tracker.update(m['topic'], emb)
    detect_time = time.time() - t0

    return _make_result(messages, predictions, diagnostics, 0.0, detect_time)


def _run_two_signal(messages, embeddings) -> BenchmarkResult:
    """Run the two-signal detector (consecutive + window must both drop)."""
    detector = TwoSignalDetector()
    predictions = []
    diagnostics = []

    t0 = time.time()
    for i, (m, emb) in enumerate(zip(messages, embeddings)):
        is_boundary, diag = detector.update(emb)
        predictions.append(is_boundary)
        diag['msg_idx'] = i
        diag['is_boundary'] = is_boundary
        diagnostics.append(diag)
    detect_time = time.time() - t0

    return _make_result(messages, predictions, diagnostics, 0.0, detect_time)


def _run_two_signal_service(messages, embeddings) -> BenchmarkResult:
    """Run the TwoSignalBoundaryService (production service with markers)."""
    from services.two_signal_boundary_service import TwoSignalBoundaryService
    from services.memory_store import MemoryStore

    store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=store):
        detector = TwoSignalBoundaryService(thread_id="benchmark")
        predictions = []
        diagnostics = []

        t0 = time.time()
        for i, (m, emb) in enumerate(zip(messages, embeddings)):
            result = detector.update(emb, m['text'])
            predictions.append(result.is_boundary)
            diagnostics.append({
                'msg_idx': i,
                'is_boundary': result.is_boundary,
                'trigger': result.trigger,
                'confidence': result.confidence,
                'consec_sim': result.consec_sim,
                'window_sim': result.window_sim,
                'marker_found': result.marker_found,
            })
        detect_time = time.time() - t0

    return _make_result(messages, predictions, diagnostics, 0.0, detect_time)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(results: Dict[str, BenchmarkResult]):
    """Print side-by-side comparison of all detectors."""
    print("\n" + "=" * 78)
    print("  TOPIC BOUNDARY DETECTION — DETECTOR COMPARISON")
    print("=" * 78)

    first = list(results.values())[0]
    print(f"\n  Messages: {first.total_messages}  |  "
          f"Boundaries: {first.total_boundaries}  |  "
          f"Continuations: {first.total_continuations}  |  "
          f"Borderline: {first.total_borderline}")

    # Summary table
    print(f"\n  {'─' * 74}")
    print(f"  {'Detector':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}  │  "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6}  │ {'Time':>7}")
    print(f"  {'':25} {'── exact ──':>18}  │  {'── tolerant ──':>18}  │")
    print(f"  {'─' * 74}")

    for name, r in results.items():
        print(f"  {name:<25} "
              f"{r.exact_precision:>6.3f} {r.exact_recall:>6.3f} {r.exact_f1:>6.3f}  │  "
              f"{r.tolerant_precision:>6.3f} {r.tolerant_recall:>6.3f} {r.tolerant_f1:>6.3f}  │ "
              f"{r.detect_time_s:>6.4f}s")

    print(f"  {'─' * 74}")

    # Per-segment comparison
    print(f"\n  {'─' * 74}")
    print(f"  PER-SEGMENT RESULTS")
    print(f"  {'─' * 74}")

    header_names = list(results.keys())
    short_names = [n[:8] for n in header_names]
    print(f"  {'Segment':<28} {'Expect':>8}  ", end='')
    for sn in short_names:
        print(f"  {sn:>8}", end='')
    print()
    print(f"  {'─' * (40 + 10 * len(results))}")

    # Iterate segments
    n_segments = len(list(results.values())[0].segment_results)
    for si in range(n_segments):
        seg_name = list(results.values())[0].segment_results[si]['topic']
        expected = list(results.values())[0].segment_results[si]['expected_boundary']
        print(f"  {seg_name:<28} {expected:>8}  ", end='')

        for name in header_names:
            sr = results[name].segment_results[si]
            fired = sr['any_fire_in_window']
            if expected == 'boundary':
                symbol = ' TP' if fired else 'MISS'
            elif expected == 'borderline':
                symbol = ' ok' if not fired else ' ok'
            else:
                symbol = ' FP!' if fired else '  ok'
            print(f"  {symbol:>8}", end='')
        print()

    print(f"  {'─' * (40 + 10 * len(results))}")

    # Missed boundaries detail for best detector
    best_name = max(results, key=lambda n: results[n].tolerant_f1)
    best = results[best_name]
    misses = [sr for sr in best.segment_results
              if sr['expected_boundary'] == 'boundary' and not sr['any_fire_in_window']]
    if misses:
        print(f"\n  MISSED BOUNDARIES ({best_name}):")
        for sr in misses:
            print(f"    [{sr['segment_idx']}] {sr['topic']} — \"{sr['first_msg'][:65]}\"")

    fps_segs = [sr for sr in best.segment_results
                if sr['expected_boundary'] == 'none' and sr.get('false_fires', 0) > 0]
    if fps_segs:
        print(f"\n  FALSE POSITIVE SEGMENTS ({best_name}):")
        for sr in fps_segs:
            print(f"    [{sr['segment_idx']}] {sr['topic']} — {sr['false_fires']} false fire(s)")

    print("\n" + "=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pytest entry point
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestTopicBoundaryBenchmark:
    """Integration benchmark — requires embedding model."""

    def test_detector_comparison(self):
        """Run all detectors on 325 messages, compare results."""
        from services.embedding_service import EmbeddingService

        messages = _build_labeled_messages()
        assert len(messages) == 325, f"Expected 325 messages, got {len(messages)}"

        # Generate embeddings once (expensive)
        emb_service = EmbeddingService()
        t0 = time.time()
        embeddings = [emb_service.generate_embedding_np(m['text']) for m in messages]
        embed_time = time.time() - t0
        print(f"\n  Embeddings: {len(embeddings)} in {embed_time:.2f}s")

        # Run all detectors
        results = {
            'Adaptive (current)': _run_adaptive(messages, embeddings),
            'Static (0.65)': _run_static(messages, embeddings, threshold=0.65),
            'Two-Signal (inline)': _run_two_signal(messages, embeddings),
            'Two-Signal Service': _run_two_signal_service(messages, embeddings),
        }

        _print_comparison(results)

        # Trigger breakdown for service
        svc = results['Two-Signal Service']
        triggers = {}
        for sr in svc.segment_results:
            for d in sr['diagnostics']:
                if d.get('is_boundary'):
                    t = d.get('trigger', 'unknown')
                    triggers[t] = triggers.get(t, 0) + 1
        if triggers:
            print("  TWO-SIGNAL SERVICE — TRIGGER BREAKDOWN:")
            for t, count in sorted(triggers.items()):
                print(f"    {t}: {count}")
            print()

        # Save results
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        report = {}
        for name, r in results.items():
            report[name] = {
                'exact': {'precision': r.exact_precision, 'recall': r.exact_recall, 'f1': r.exact_f1},
                'tolerant': {'precision': r.tolerant_precision, 'recall': r.tolerant_recall, 'f1': r.tolerant_f1},
                'segments': r.segment_results,
            }
        report_path = output_dir / "topic_boundary_comparison.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  Results saved to: {report_path}")

        # Assert: service must beat current adaptive detector
        svc_result = results['Two-Signal Service']
        adaptive = results['Adaptive (current)']
        assert svc_result.tolerant_f1 > adaptive.tolerant_f1, (
            f"Two-Signal Service F1 ({svc_result.tolerant_f1:.3f}) must beat "
            f"Adaptive F1 ({adaptive.tolerant_f1:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    logging.basicConfig(level=logging.WARNING)

    from services.embedding_service import EmbeddingService

    messages = _build_labeled_messages()
    emb_service = EmbeddingService()

    print(f"Generating {len(messages)} embeddings...")
    t0 = time.time()
    embeddings = [emb_service.generate_embedding_np(m['text']) for m in messages]
    print(f"Done in {time.time() - t0:.2f}s")

    results = {
        'Adaptive (current)': _run_adaptive(messages, embeddings),
        'Static (0.65)': _run_static(messages, embeddings, threshold=0.65),
        'Two-Signal (inline)': _run_two_signal(messages, embeddings),
        'Two-Signal Service': _run_two_signal_service(messages, embeddings),
    }
    _print_comparison(results)
