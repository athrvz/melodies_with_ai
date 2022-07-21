import tensorflow.keras as keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import music21 as m21 


class MelodyGenerator:

	def __init__(self, model_path = "model.h5"):

		self.model_path = model_path
		self.model = keras.models.load_model(model_path)

		with open(MAPPING_PATH, "r") as fp:
			self._mappings = json.load(fp)

		self._start_symbols = ["/"] * SEQUENCE_LENGTH

	def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

		# create seed with start symbols
		seed = seed.split()
		melody = seed
		seed = self._start_symbols + seed

		# map seed to int
		seed = [self._mappings[symbol] for symbol in seed]

		for _ in range(num_steps):

			# limit seed to max_sequence_len
			seed = seed[-max_sequence_length:]

			# one-hot encode the seed
			onehot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings)) 
			# (1, max_sequence_length, num of symbols in vocab)
			onehot_seed = onehot_seed[np.newaxis, ...]

			# make a prediction
			probabilities = self.model.predict(onehot_seed)[0]

			output_int = self._sample_with_temperature(probabilities, temperature)

			# update seed
			seed.append(output_int)

			# mapping
			ouput_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

			# check whether we're at the end of melody
			if ouput_symbol == "/":
				break

			# update melody
			melody.append(ouput_symbol)

		return melody


	def _sample_with_temperature(self, probabilities, temperature):
		predictions = np.log(probabilities) / temperature
		probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

		choices = range(len(probabilities)) 
		index = np.random.choice(choices, p = probabilities)

		return index


	def save_melody(self, melody, step_duration = 0.25, format = "midi", file_name = "mel.mid"):

		# create a music21 stream
		stream = m21.stream.Stream()

		# parse all symbols and create note, rests obj
		start_symbol = None
		step_counter = 1

		for i, symbol in enumerate (melody):

			# handle case in which we ahve a note/ rest
			if symbol != "_" or i + 1 == len(melody):
				if start_symbol is not None:

					quarter_length_duration = step_duration * step_counter

					# handle rest
					if start_symbol == "r":
						m21_event = m21.note.Rest(quarterLength = quarter_length_duration)
					# handle note
					else:
						m21_event = m21.note.Note(int(start_symbol), quarterLength = quarter_length_duration)

					stream.append(m21_event)

					# reset the step_counter
					step_counter = 1

				start_symbol = symbol

			# handle case in which we have a prolongation sign "_"
			else:
				step_counter += 1
		# write the m21 stream to a midi file
		stream.write(format, file_name)

if __name__ == "__main__":
	mg = MelodyGenerator()
	seed = "55 _ 60 _ 60 _ "
	melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.4)
	print(melody)
	mg.save_melody(melody)