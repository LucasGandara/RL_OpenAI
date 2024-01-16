import random

import matplotlib
import matplotlib.pyplot as plt


def roll_dice():
    roll = random.randint(1, 100)
    if roll == 100:
        # print(f"Roll was 100, you lose! Critical Fail")
        return False
    elif roll <= 50:
        # print(f"Roll was {roll}, you lose!")
        return False
    elif 100 > roll >= 50:
        # print(f"Roll was {roll}, you win!")
        return True


def simple_betor(funds, initial_wager, wager_count):
    value = funds
    wager = initial_wager

    wX = []
    vY = []

    currentWager = 1
    while currentWager <= wager_count:
        if roll_dice():
            value += wager
            wX.append(currentWager)
            vY.append(value)
        else:
            value -= wager
            wX.append(currentWager)
            vY.append(value)

        currentWager += 1

    if value < 0:
        value = "Broke"

    print(f"Funds: {value}")
    plt.plot(wX, vY)


def main():
    n = 0
    while n < 1000:
        simple_betor(10000, 100, 100000)
        n += 1

    plt.ylabel("Account Value")
    plt.xlabel("Wager Count")
    plt.show()


if __name__ == "__main__":
    main()
