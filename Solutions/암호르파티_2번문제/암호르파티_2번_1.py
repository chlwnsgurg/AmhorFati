# Model import
from keras.models import model_from_json
arch = open('arch_neural_distinguisher.json')
json_arch = arch.read()
nr6_speck_distinguisher = model_from_json(json_arch)
nr6_speck_distinguisher.load_weights('weights_nr6_speck.h5')
nr6_speck_distinguisher.compile(optimizer='adam',loss='mse',metrics=['acc'])

# Model test
import speck as sp
size = 10 ** 6
x_test, y_test = sp.real_differences_data(size, 6)
print(x_test)
#x_test, y_test = sp.make_train_data(size, 6)
res = nr6_speck_distinguisher.predict(x_test)

correct = 0
for i, j in zip(res, y_test):
    if round(float(i[0])) == j:
        correct += 1

results = nr6_speck_distinguisher.evaluate(x_test, y_test, batch_size=10000)
print('test loss, test_acc: ', results)
print('Calculated acc: ', correct / size)

if round(results[1], 6) == correct / size:
    print("The result of distinguisher is the prob of Y_label = 1")
