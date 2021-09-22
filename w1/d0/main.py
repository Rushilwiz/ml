import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.linspace(0, 20, 100)
    plt.plot(x, np.sin(x))
    plt.xlabel("x")
    plt.xlabel("sin(x)")
    plt.title("A sine curve")
    plt.show()

if __name__ == "__main__":
    main()
