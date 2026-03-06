from collections import deque
import time


class WordBuilder:

    def __init__(self):
        self.letter_buffer = deque(maxlen=10)

        self.current_letter = None
        self.current_word = ""

        self.last_letter_time = time.time()

        # tuning parameters
        self.LETTER_STABLE_COUNT = 5
        self.LETTER_COOLDOWN = 0.8
        self.WORD_PAUSE = 1.5


    def update(self, letter):
        """
        Update builder with a new predicted letter.
        Returns:
            (current_word, new_letter_added)
        """

        now = time.time()

        if letter == "nothing":
            # detect pause to finalize word
            if now - self.last_letter_time > self.WORD_PAUSE:
                return self.current_word, False
            return self.current_word, False

        self.letter_buffer.append(letter)

        # check if last N predictions agree
        if len(self.letter_buffer) >= self.LETTER_STABLE_COUNT:

            recent = list(self.letter_buffer)[-self.LETTER_STABLE_COUNT:]

            if len(set(recent)) == 1:
                stable_letter = recent[0]

                # avoid duplicate hold
                if stable_letter != self.current_letter:

                    if now - self.last_letter_time > self.LETTER_COOLDOWN:

                        self.current_letter = stable_letter
                        self.last_letter_time = now

                        if stable_letter == "space":
                            self.current_word += " "

                        elif stable_letter == "del":
                            self.current_word = self.current_word[:-1]

                        else:
                            self.current_word += stable_letter

                        return self.current_word, True

        return self.current_word, False