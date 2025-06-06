"""Chapter 1: Introduction to NLP."""

import re
from collections import Counter
from itertools import permutations

greetings = "Hi Hello Greetings".split()
user_statement = "Hello Joshua"
user_token_sequence = user_statement.split()
print(user_token_sequence)

if user_token_sequence[0] in greetings:
    bot_reply = "Thermonuclear War is a strange game. "
    bot_reply += "The only winning move is not to play."
else:
    bot_reply = "Would you like to play a nice game of chess?"

print(bot_reply)

r = "(hi|hello|hey|)[ ,:.!]*([a-z]*)"

print(re.match(r, "hi ho, hi ho, it's off to work ...", flags=re.IGNORECASE))
print(re.match(r, "Hello Rosa", flags=re.IGNORECASE))
print(re.match(r, "hey, what's up?", flags=re.IGNORECASE))

r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])(morn[gin']{0,3}|"
r += r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
print(re_greeting.match("Hello Rosa"))
print(re_greeting.match("Hello Rosa"))
print(re_greeting.match("Hello Rosa").groups())  # type: ignore
print(re_greeting.match("Good morning Rosa"))
print(re_greeting.match("Good Manning Rosa"))
print(re_greeting.match("Good evening Rosa Parks").groups())  # type: ignore
print(re_greeting.match("Good Morn'n Rosa"))
print(re_greeting.match("yo Rosa"))

# my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot','chatterbot'])
# curt_names = set(["hal", "you", "u"])
# greeter_name = ""
# wd = "you"
# match = re_greeting.match(input())
# if match:
#     at_name = match.groups()[-1]
#     if at_name in curt_names:
#         breakpoint()
#         print("Good one.")
#     elif at_name.lower() in my_names:
#         print(f"Hi {greeter_name}, how are you?")
#     else:
#         print("Fu.")

# print(greeter_name)


print(Counter(user_token_sequence))
print(Counter("Guten Morgen Rosa".split()))
print(Counter("Guten Morgen Rosa!".split()))

gram = [" ".join(combo) for combo in permutations("Good morning Rosa!".split(), 3)]
print(gram)

gilles_regex = re.compile(r"(gilles|gilou)(?:\s+)(major)?", flags=re.IGNORECASE)
print(gilles_regex.match("Gilles"))
print(gilles_regex.match("gilles"))
print(gilles_regex.match("giLLes "))
print(gilles_regex.match("Gillles"))
print(gilles_regex.match("Gilles major"))
print(gilles_regex.match("Gilles majoR"))
print(gilles_regex.match("Gilles maJor"))
print(gilles_regex.match("Gilles  maJor"))
# print(gilles_regex.match("Gilles major").groups())


the_sentence = "Gilles is a major. Gilou is a major."
advanced_regex = r"(?<!\w)[.!?](?!\w)"

# Using match() - only looks at the beginning of the string
print("Using match():", re.match(advanced_regex, the_sentence))  # None

# Using search() - looks anywhere in the string
print("Using search():", re.search(advanced_regex, the_sentence))  # Finds a match

# To find all matches, use findall()
print(
    "Using findall():", re.findall(advanced_regex, the_sentence)
)  # ['.', '.']the_sentence = "Gilles is a major. Gilou is a major."
lookbehind_regex = r"(?<=[a-z])[.!?](?=\s|$)"
print(re.search(lookbehind_regex, the_sentence))
print(re.search(lookbehind_regex, "Gilles is a major. Gilou is a major."))
print(re.search(lookbehind_regex, "Dr. Gilles is a major. Gilou is a major."))
