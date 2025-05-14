epochs = 100
l_rate = 0.001

layers = [
    LayerSetup(512, 'relu'),
    LayerSetup(256, 'relu'),
    LayerSetup(128, 'relu'),
    LayerSetup(5, 'softmax'),
]

datasets_names = ['car.bin',]
# 'crab.bin', 'duck.bin', 'ice_cream.bin', 'sock.bin']

def get_path(file: str) -> str:
    return os.path.join(os.getenv('DATASET_ADDRESS', ''), file)

x = []
y = []

for i, dt in enumerate(datasets_names):
    print('-='*15)
    for drawing_num, drawing in enumerate(unpack_drawings(get_path(dt))):
        image = np.array(drawing['image'][0], dtype=np.float64) / 255.0
        x.append(image)
        y.append(i)
        print(f"{dt} número {drawing_num} adicionado!")

network = NeuralNetwork.new(layers, x[0].shape)
training = Training(0.01, network)

x = np.array(x, dtype=np.float64)
y = np.array(y,  dtype=np.float64)
training.train(x, y, 60, 16)
