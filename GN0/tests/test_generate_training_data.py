from GN0.generate_training_data import generate_graphs

def test_generate_training_data():
    graphs = generate_graphs(1)

    print(graphs)


if __name__ == '__main__':
    test_generate_training_data()