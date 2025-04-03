"""Chapter 1: Introduction to NLP."""

import re

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
print(re_greeting.match("Hello Rosa").groups())  # type: ignore
print(re_greeting.match("Good morning Rosa"))
print(re_greeting.match("Good Manning Rosa"))
print(re_greeting.match("Good evening Rosa Parks").groups())  # type: ignore
print(re_greeting.match("Good Morn'n Rosa"))
print(re_greeting.match("yo Rosa"))
