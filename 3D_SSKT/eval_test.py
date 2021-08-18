import json

if __name__ == '__main__':
f = open('val.json')
data = json.load(f)
total = 0.0
count = 0.0
for video_name in data['results'].keys():
    if data['results'][video_name][0]['label'].lower() in video_name.lower():
        count += 1
    total += 1

print('accuracy: ', count / total)