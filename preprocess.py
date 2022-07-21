import os
import json
import music21 as m21
import numpy as np 
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"

ACCEPTABLE_DURATIONS = [
	0.25, 
	0.5,
	0.75,
	1.0,
	1.5,
	2,
	3, 
	4
]
def load_songs_in_kern(dataset_path):

	songs = []
	# go through all files in dataset and load with music21
	for path, subdirs, files, in os.walk(dataset_path):
		for file in files:
			if file[-3:] == "krn":
				song = m21.converter.parse(os.path.join(path, file))
				songs.append(song)
	return songs

def has_acceptable_duration(song, acceptable_duration):
	for note in song.flat.notesAndRests:
		if note.duration.quarterLength not in acceptable_duration:
			return False
	return True


def transpose(song):
	# get key from the song
	parts = song.getElementsByClass(m21.stream.Part)
	measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
	key = measures_part0[0][4]

	# estimate key using m21
	if not isinstance(key, m21.key.Key):
		key = song.analyze("key")

	print(f"Key: {key}")

	# get interval for transposition
	if key.mode == "major":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))

	elif key.mode == "minor":
		interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

	# transpose song by calculated interval
	transposed_song = song.transpose(interval)
	return transposed_song

def encode_song(song, time_step = 0.25):
	# p = 60, d = 1.0 -> [60, "_", "_", "_", "_"]
	encoded_song = []

	for event in song.flat.notesAndRests:
		# handle notes
		if isinstance(event, m21.note.Note):
			symbol = event.pitch.midi
		elif isinstance(event, m21.note.Rest):
			symbol = "r"

		# convert the note / rest into time series notation
		steps = int(event.duration.quarterLength / time_step)
		for step in range(steps):
			if step == 0:
				encoded_song.append(symbol)
			else:
				encoded_song.append("_")

	# cast encoded song to str
	encoded_song = " ".join(map(str, encoded_song))

	return encoded_song


def preprocess(dataset_path):
	pass

	# load the folk songs
	print("Loading songs....")
	songs = load_songs_in_kern(dataset_path)
	print(f"Loaded {len(songs)} songs.")

	for i, song in enumerate(songs):
		# filter out songs that have non-acceptable durations
		if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
			continue

		# transpose songs to Cmaj or Amin
		song = transpose(song)

		# encode songs with time series representation
		encoded_song = encode_song(song)

		# save songs to txt file
		save_path = os.path.join(SAVE_DIR, str(i))
		with open(save_path, "w") as fp:
			fp.write(encoded_song)

def load(file_path):
	with open(file_path, "r") as fp:
		song = fp.read()
		return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
	new_song_delimiter = "/ " * sequence_length
	songs = ""

	# load encoded songs and add delimiters
	for path, _, files in os.walk(dataset_path):
		for file in files:
			file_path = os.path.join(path, file)
			song = load(file_path)
			songs = songs + song + " " + new_song_delimiter

	songs = songs[:-1]

	# save str that contains all dataset
	with open(file_dataset_path, "w") as fp:
		fp.write(songs)
	return songs

def create_mapping(songs, mapping_path):
	mappings = {}
	# identify vocab
	songs = songs.split()
	vocabulary = list(set(songs))

	# create mapping
	for i, symbol in enumerate(vocabulary):
		mappings[symbol] = i

	# save vocab to a json file
	with open(mapping_path, "w") as fp:
		json.dump(mappings, fp, indent = 4)


def convert_songs_to_int(songs):
	int_songs = []

	# load mappings
	with open(MAPPING_PATH, "r") as fp:
		mappings = json.load(fp)

	# cast songs str to a list
	songs = songs.split()

	# map songs to int
	for symbol in songs:
		int_songs.append(mappings[symbol])

	return int_songs


def generate_training_sequences(sequence_length):

	# [11, 12, 13, ....] -> input: [11, 12,] , targets: 13; input: [12, 13], targets: 14

	# load songs and convert to int
	songs = load(SINGLE_FILE_DATASET)
	int_songs = convert_songs_to_int(songs)

	# generate training sequ
	# 100 symbols, sequ len = 64, sequences = 100 - 64 = 36
	inputs = []
	targets = []
	num_sequences = len(int_songs) - sequence_length
	for i in range(num_sequences):
		inputs.append(int_songs[i: i+sequence_length])
		targets.append(int_songs[i+sequence_length])
	
	# one-hot encode the sequ 
	# inputs: (no of sequ, sequ len, vocab size)
	# [ [0, 1, 2], [1, 1, 2] ] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1], []]
	vocabulary_size = len(set(int_songs))
	inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
	targets = np.array(targets)

	return inputs, targets

def main():
	# songs = load_songs_in_kern(KERN_DATASET_PATH)
	# print(f"Loaded {len(songs)} songs.")
	# song = songs[0]

	# preprocess(KERN_DATASET_PATH)
	# # print(f"has_acceptable_duration? {has_acceptable_duration(song, ACCEPTABLE_DURATIONS)}")
	
	# transposed_song = transpose(song)
	# # song.show()

	preprocess(KERN_DATASET_PATH)
	songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
	create_mapping(songs, MAPPING_PATH)
	# inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
	main()