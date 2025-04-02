"""Chapter 1: Introduction to NLP."""

import re

greetings = " Hi Hello Greetings".split()
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
