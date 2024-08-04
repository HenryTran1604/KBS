import numpy as np
import librosa, os
from service import extract_features
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

X = []
y = []
label_mapping = ["Type A", "Type B", "Type C", "Type D"]
for user_dir in os.listdir('data/speakers'):
    user_path = os.path.join('data/speakers', user_dir)
    if os.path.isdir(user_path):
        # Iterate through each type directory
        for type_dir in label_mapping:
            type_path = os.path.join(user_path, type_dir)
            if os.path.isdir(type_path):
                # Iterate through each file in the type directory
                for file_name in os.listdir(type_path):
                    file_path = os.path.join(type_path, file_name)
                    if os.path.isfile(file_path) and file_path.endswith('.wav'):
                        # Load the audio file
                        mfcc_mean = extract_features(file_path=file_path)
                        X.append(mfcc_mean)
                        y.append(1)
print(len(X))
for (_, _, f) in os.walk('data/AI'):
    for file in f:
        file_path = os.path.join('data/AI', file)
        mfcc_mean = extract_features(file_path=file_path)
        X.append(mfcc_mean)
        y.append(0)



X = np.array(X)
y = np.array(y)  # 1 cho giọng thật, 0 cho giọng AI

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
joblib.dump(model, 'fake_voice.pkl')

# # # Dự đoán trên tệp âm thanh mới
# # new_file = 'path/to/new_file.wav'
# # new_features = extract_features(new_file).reshape(1, -1)
# # prediction = model.predict(new_features)
# # print('Real Voice' if prediction[0] == 0 else 'AI Voice')
