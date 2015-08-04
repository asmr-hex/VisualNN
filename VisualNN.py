# VisualNN
# A didactic tool for visualizing simple artificial neural networks and 
# how they learn.
#
# MIT © tuntun, July 2015 

class NeuralNetwork(object):
	"""A general neural network supporting a variety of architectures. 
	
	The network class provides the architecture of configured neurons 
	as well as the processes with which to operate on them.
	"""
	
	__Layers = []		#Neuronal layers
	__Weights = []		#Network Weights
	def __init__(self):
		"""Inits NeuralNetwork with architecture."""
		pass
		
	def feedForward(self, x):
		"""Feeds network inputs, stores intermediate values, returns output."""
		pass
		
	def backPropagation(self):
		"""Performs backpropogation of errors over the neural network."""
		pass
		
		
class Neuron(object):
	"""A single computational neuron.
	
	The neuron class provides an activation function for inputs, 
	storage for activations achieved during forward passes, and
	storage for error signals during backwards propogation. """
	
	__f = ""		#Activation function
	__A = 0			#Current activation
	__Links = []	#Directed edgelist
	def __init__(self):
		"""Inits Neuron with activation function."""
		pass
		
	def computeActivation(self, x):
		"""Uses activation function to compute activation."""
		pass
		
	def addLink(self, neuron, direction):
		"""Adds directed connectivity between this and another neuron."""
		pass
		
		
class Illustration(object):
	"""Illustration tool for visualizing neural networks.
	
	Instantiated within a NeuralNetwork object and called from within."""
	
	def __init__(self):
		"""Inits visualization tool for a NeuralNetwork instance."""
		pass
		
	def draw(self):
		"""Draws the current NeuralNetwork instance state."""
		pass
		
		
		
		
		
	
	
	