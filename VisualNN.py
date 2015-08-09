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
	__Weights = []	#Network Weights
	__illustrator = None	#Illustrator instance
	def __init__(self, architecture):
		"""Inits NeuralNetwork with architecture."""
		self.__buildArchitecture(architecture)
		self.__illustrator = Illustrator(self.__Layers)
		
	def __buildArchitecture(self, architecture):
		"""Parses architecture string and builds network."""
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
					
	def __parseRange(self, start, end):
		"""Helper function to parse ranges for architecture setup"""
		end = int(end)
		if start is None:
			start = end
		else:
			start = int(start)
		return range(start, end+1)
		
	def __getNeurons(self, layer, nids):
		"""Helper function to find neuron, or return a new one"""
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
	
	__f = ""	#Activation function
	__A = 0	#Current activation
	__inputs = []	#list of input neurons
	__outputs = []	#list of output neurons
	__layer = None
	__id = None
	__x = 0	# graphical x positioning
	__y = 0	# graphical y positioning
	__r = 0 # graphical radius
	__c = 0	# graphical color
	__graphic = None	#graphical representation
	def __init__(self, layer, nid):
		"""Inits Neuron with activation function."""
		self.__layer = layer
		self.__id = nid
		
	def computeActivation(self, x):
		"""Uses activation function to compute activation."""
		pass
	
	def getActivation(self):
		"""Gets current activation from neuron."""
		return self.__A
		
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
		
	def getID(self):
		"""Get the neuron id of this neuron."""
		return self.__id
	
	def getLayer(self):
		"""Get the layer in which this neuron resides."""
		return self.__layer
		
	def setXYR(self, x, y, r):
		"""Sets the graphical x,y positions."""
		self.__x = x
		self.__y = y
		self.__r = r
		
	def setColor(self, color):
		"""Set graphical color."""
		self.__c = color
		
	def getGraphic(self):
		"""Returns graphical object for this neuron."""
		circle = plt.Circle((self.__x, self.__y),
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
				self.__neurons[l].append(layers[l][n].getGraphic())
				self.__ax.add_patch(self.__neurons[l][n])
				edges = layers[l][n].getOutput()
				for e in xrange(0, len(edges)):
					dest_l = edges[e].getLayer()
					dest_height = (self.__ax_ylim)/(len(layers[dest_l]))
					edgex = [x, self.__layer_x[dest_l]]
					edgey = [y, e*dest_height + (dest_height/2)]
					edgeStr = 'L'+str(l)+'N'+str(n)+'->'
					edgeStr+= 'L'+str(dest_l)+'N'+str(edges[e].getID())
					self.__edges[edgeStr] = [[x,y],[edgex[1],edgey[1]]]
					self.__ax.plot(edgex, edgey, color='k', zorder=1)
		self.__ax.set_ylim(0,self.__ax_ylim)
		self.__ax.set_xlim(0,self.__ax_xlim)
		self.__ax.axis('scaled')
		self.__ax.axis('Off')
		plt.show()
		
	def draw(self):
		"""Draws the current NeuralNetwork instance state."""
		pass
		