import pickle

# Correct usage of 'with' and 'open'
with open("metrics0.p", 'rb') as file:
    data = pickle.load(file)

print(data['S'][0])
