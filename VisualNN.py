# VisualNN
# A didactic tool for visualizing simple artificial neural networks and 
# how they learn.
#
# MIT copyright tuntun, July 2015 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import re

class NeuralNetwork(object):
	"""A general neural network supporting a variety of architectures. 
	
	The network class provides the architecture of configured neurons 
	as well as the processes with which to operate on them.
	"""
	
	__Layers = {}	#Neuronal layers
	__illustrator = None	#Illustrator instance
	Error = 0	#Output error
	
	def __init__(self, architecture, f, df, verbose=False):
		"""Inits NeuralNetwork with architecture."""
		self.__verbose = verbose
		self.__buildArchitecture(architecture)
		self.__initializeWeights()
		self.__setActivationFunctions(f,df)
		self.__illustrator = Illustrator(self.__Layers)
		
	def __buildArchitecture(self, architecture):
		"""Parses architecture string and builds network."""
		if self.__verbose:
			print "Building NN Architecture..."
		L_GROUP = 1	#Layer group for regex
		S_GROUP = 4	#NodeID start group for regex
		E_GROUP = 5	#NodeID end group for regex
		chains = re.split(r',(?=\(\d+,\d*:?\d+\))', architecture)
		rexp = re.compile(r'\((\d+),(((\d*):)?(\d*))\)')
		for chain in chains:	#Parse each defined chain
			chain = re.split(r'->',chain)
			for i in xrange(0,len(chain)-1):
				src_node = rexp.search(chain[i])	#Get source neuron(s)
				src_layer = int(src_node.group(L_GROUP))
				src_ids = self.__parseRange(src_node.group(S_GROUP),src_node.group(E_GROUP))
				src_neurons = self.__getNeurons(src_layer, src_ids)
				dest_node = rexp.search(chain[i+1])	#Get destination neuron(s)
				dest_layer = int(dest_node.group(L_GROUP))
				dest_ids = self.__parseRange(dest_node.group(S_GROUP),dest_node.group(E_GROUP))
				dest_neurons = self.__getNeurons(dest_layer, dest_ids)
				for src_neuron in src_neurons:
					src_neuron.addOutput(dest_neurons)	#Add dests to src
				for dest_neuron in dest_neurons:
					dest_neuron.addInput(src_neurons)	#Add src to dests
	
	def illustrate(self):
		"""Illustrate network design."""
		self.__illustrator.draw()
					
	def __parseRange(self, start, end):
		"""Helper function to parse ranges for architecture setup."""
		end = int(end)
		if start is None:
			start = end
		else:
			start = int(start)
		return range(start, end+1)
		
	def __getNeurons(self, layer, nids):
		"""Helper function to find neuron, or return a new one."""
		neurons = []
		for nid in nids:
			layers = self.__Layers.keys()
			if layer not in layers:
				neuron = Neuron(layer, nid)
				self.__Layers[layer] = [neuron]
				neurons.append(neuron)
			else:
				neuron_ids = list(map((lambda x: x.getID()),self.__Layers[layer]))
				if nid in neuron_ids:
					neurons.append(self.__Layers[layer][neuron_ids.index(nid)])
				else:
					neuron = Neuron(layer, nid)
					self.__Layers[layer].append(neuron)
					neurons.append(neuron)
		return neurons
	
	def __initializeWeights(self):
		"""Initialize weights and biases."""
		if self.__verbose:
			print "Initializing NN Weights and Biases..."
		for L in xrange(0,len(self.__Layers)-1):
			#Initialize one bias for each layer
			bias = Neuron(L, len(self.__Layers[L]), bias=True)
			bias.addOutput(self.__Layers[L+1])
			for neuron in self.__Layers[L+1]:
				neuron.addInput([bias])
			#Initialize weights for this layer
			self.__Layers[L].append(bias)
			for neuron in self.__Layers[L]:
				outward = neuron.getOutput()
				for next_neuron in outward:
					neuron.w[next_neuron.getID()] = np.random.rand(1)
	
	def __setActivationFunctions(self, f, df):
		"""Assign Activation functions and derivatives to neurons."""
		#Handle case if there are multiple types of Activations
		if type(f) is list and type(df) is list:
			activation_type = 'multiple'
			if len(f) != len(self.__Layers) or len(df) != len(self.__Layers):
				raise ValueError('number of activations and layers must equal!')
		else:
			activation_type = 'single'
		for L in self.__Layers.keys():
			#Label layer type
			if L == 0:
				layer_type = "input"
			elif L == len(self.__Layers.keys())-1:
				layer_type = "output"
			else:
				layer_type = "hidden"
			#Select function
			if activation_type is 'single':
				F = f
				dF = df
			else:
				F = f[L]
				dF = df[L]
			for neuron in self.__Layers[L]:
				neuron.setLayerType(layer_type)
				neuron.setFunctions(F, dF)
		
	def getOutput(self):
		"""Get output layer of network."""
		L = len(self.__Layers) - 1
		output = np.empty([1,len(self.__Layers[L])])
		for idx,neuron in enumerate(self.__Layers[L]):
			output[0,idx] = neuron.A
		return output
		
	def feedForward(self, x):
		"""Feeds network inputs, stores intermediate values, returns output."""
		if self.__verbose:
			print "Feeding Forward..."
		#Ensure that input size matches inputs
		if len(x) != len(self.__Layers[0])-1:
			raise ValueError("incorrect network input size!")
		for L in self.__Layers.keys():
			idx = 0
			for neuron in self.__Layers[L]:
				#Set input neurons to input data
				if neuron.layer_type is "input":
					if neuron.bias:		#Don't set biases in input layer
						continue
					else:
						neuron.A = x[idx]
						idx+=1
						continue
				#Compute Activations for each neuron
				if neuron.bias:
					continue
				else:
					for inputs in neuron.getInput():
						neuron.Sigma += inputs.w[neuron.getID()] * inputs.A
					neuron.computeActivation(neuron.Sigma)
		
	def backPropagation(self, y):
		"""Performs backpropogation of errors over the neural network."""
		if self.__verbose:
			print "backpropagating errors..."
		self.computeError(y)
		L = len(self.__Layers.keys()) - 1
		#Propogate error signal backwards through the network
		for Lidx in xrange(L, 0, -1):
			layer = self.__Layers[Lidx]
			for idx, neuron in enumerate(layer):
				if neuron.layer_type is 'output':
					neuron.dError = self.computeErrorPrime(y[idx], neuron.A)
				else:
					neuron.dError = 0	#Clear error signal for SGD (TODO: support Batch training?)
					for outputs in neuron.getOutput():
						neuron.dError += outputs.dError * neuron.w[outputs.getID()]
						
	def updateWeights(self, rate, momentum=1):
		"""Update the weights based on the propogated error signal."""
		if self.__verbose:
			print "updating weights..."
		for L in self.__Layers.keys():
			for neuron in self.__Layers[L]:
				outputs = neuron.getOutput()
				#update weights
				for out in outputs:
					delta = rate * out.dError * out.dF(out.Sigma) * neuron.A
					neuron.w[out.getID()] +=  delta
						
		
	def computeError(self, y):
		"""Compute error signal from outputs."""
		o = self.getOutput()
		if len(o) != len(y):
			raise ValueError("Size mismatch between labels and outputs!")
		self.Error = 0.5 * np.sum([(y[i]-o[i])**2 for i in range(0,len(y))])
		
	def computeErrorPrime(self, yi, ai):
		"""Compute the derivative of the error signal."""
		return -(yi-ai)
		
		
class Neuron(object):
	"""A single computational neuron.
	
	The neuron class provides an activation function for inputs, 
	storage for activations achieved during forward passes, and
	storage for error signals during backwards propogation. """
	
	__f = None	#Activation function
	__df = None	#Derivative of Activation function
	A = 0.	#Current activation
	dError = 0	#Propogated error signal
	Sigma = 0.	#Current Summation
	bias = False	#is this a bias node?
	__inputs = []	#list of input neurons
	__outputs = []	#list of output neurons
	w = {}	#Dictionary of weights corresponding to outputs
	__layer = None
	__id = None
	x = 0	# graphical x positioning
	y = 0	# graphical y positioning
	__r = 0 # graphical radius
	__c = 0	# graphical color
	__graphic = None	#graphical representation
	def __init__(self, layer, nid, bias=False):
		"""Inits Neuron with activation function."""
		self.__layer = layer
		self.__id = nid
		if bias:
			self.bias = True
			self.A = 1
			self.Sigma = 1
	
	def setLayerType(self, layer_type):
		"""Set the type of layer this neuron resides in."""
		self.layer_type = layer_type
		
	def setFunctions(self, f, df):
		"""Sets activation function and derivative."""
		self.__f = f
		self.__df = df
		
	def computeActivation(self, x):
		"""Uses activation function to compute activation."""
		self.A = self.__f(x)
		
	def dF(self, x):
		"""Computes the derivative of the activation function."""
		return self.__df(x)
		
	def addOutput(self, neurons):
		"""Adds output connectivity between this and other neurons."""
		if type(neurons) is list:
			self.__outputs =  self.__outputs + neurons
		else:
			self.__outputs.append(neurons)
		
	def addInput(self, neurons):
		"""Adds input connectivity between this and other neurons."""
		if type(neurons) is list:
			self.__inputs = self.__inputs + neurons
		else:
			self.__inputs.append(neurons)
			
	def getOutput(self):
		"""Gets output connected neurons."""
		return self.__outputs
		
	def getInput(self):
		"""Gets input connected neurons."""
		return self.__inputs
		
	def getID(self):
		"""Get the neuron id of this neuron."""
		return self.__id
	
	def getLayer(self):
		"""Get the layer in which this neuron resides."""
		return self.__layer
		
	def setXYR(self, x, y, r):
		"""Sets the graphical x,y positions."""
		self.x = x
		self.y = y
		self.__r = r
		
	def setColor(self, color):
		"""Set graphical color."""
		self.__c = color
		
	def getGraphic(self):
		"""Returns graphical object for this neuron."""
		circle = plt.Circle((self.x, self.y),
		 			radius=self.__r, color=self.__c, zorder=2,
					clip_on=False)
		return circle
		
		
class Illustrator(object):
	"""Illustration tool for visualizing neural networks.
	
	Instantiated within a NeuralNetwork object and called from within."""
	
	__on = False	#Illustrator on/off
	__fig = plt.figure()	#Illustrator figure handles
	__ax = __fig.gca()		#Illustrator axes
	__ax_xlim = 0	# x limits of network axis
	__ax_ylim = 0	# y limits of network axis
	__radii = 1	# graphical neuron radii
	__neurons = {}			#Graphical neurons
	__edges = {}			#graphical edges
	__layer_colors = []		#Layer color
	
	
	def __init__(self, layers):
		"""Inits visualization tool for a NeuralNetwork instance."""
		self.__configureArchitecture(layers)
		#self.draw()
		
		#self.__neurons[0][0].setColor("#F3CA7E")
		#plt.pause(1)   
		#self.draw() 
		
		
	def __configureArchitecture(self, layers):
		"""Configures the layout of the graphical network model."""
		layerWidth = self.__radii * 6
		layerHeight = self.__radii * 3
		#Identify the maximum number of nodes per layer across all layers
		maxNodes = max([len(layers[k]) for k in layers.keys()])
		nLayers = len(layers)
		self.__ax_xlim = nLayers*layerWidth
		self.__ax_ylim = maxNodes*layerHeight
		self.__layer_x = np.linspace(0, self.__ax_xlim, nLayers)
		for l in xrange(0,nLayers):
			height = (self.__ax_ylim)/(len(layers[l]))
			self.__neurons[l] = []
			for n in xrange(0, len(layers[l])):
				x = self.__layer_x[l]
				y = n*height + (height/2)
				layers[l][n].setXYR(x,y, self.__radii)
				layers[l][n].setColor('#2481F3')
				self.__neurons[l].append(layers[l][n])
		
	def setLayerColors(self, colors):
		"""Set colors of each layer."""
		
		
	def draw(self):
		"""Draws the current NeuralNetwork instance state."""
		plt.ion()
		plt.cla()
		for layer in self.__neurons.keys():
			for neuron in self.__neurons[layer]:
				self.__ax.add_patch(neuron.getGraphic())
				edges = neuron.getOutput()
				for next_neuron in edges:
					x = [neuron.x, next_neuron.x]
					y = [neuron.y, next_neuron.y]
					self.__ax.plot(x, y, color='k', zorder=1)
		self.__ax.set_ylim(0,self.__ax_ylim)
		self.__ax.set_xlim(0,self.__ax_xlim)
		self.__ax.axis('scaled')
		self.__ax.axis('Off')
		plt.draw()
		plt.show(block=True)
					
		