from network import *

def main():
	main_network = Network((2,10,10,1))
	number_of_batches = 4
	for i in range(number_of_batches): # i'm probably gonna wake up tmrw wondering what this is
		main_network.feedforward()
		main_network.backpropagate()