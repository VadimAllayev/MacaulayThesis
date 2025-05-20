import copy
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

class Node:
    def __init__(self, node_id, position, 
                 max_energy=1000.0,
                 resend_threshold=0,
                 transmission_range=50.0,
                 detection_range=30.0,
                 node_type="CC1310"):
        """
        Initialize a node in the P2P network with heterogeneous variations.
        
        Parameters:
        -----------
        node_id : int
            Unique identifier for the node
        position : tuple(float, float)
            2D position coordinates (x, y)
        max_energy : float
            Maximum energy/battery capacity, in Joules
        resend_threshold : int
            Upon a failed message, the maximum amount of retries a node is willing to make per generation (the maximum message size the node is willing to resend, in bytes?)
        transmission_range : float
            Maximum distance for direct communication, in meters
        detection_range : float
            Maximum distance to monitor temperature, in meters
        node_type : str
            Name of MCU, determines device attributes (voltage, current)
        """
        self.node_id = node_id
        self.position = position
        self.energy = max_energy
        self.max_energy = max_energy
        self.resend_threshold = resend_threshold
        self.transmission_range = transmission_range
        self.detection_range = detection_range
        self.node_type = node_type
        self.model_version = 0 # how many generation iterations the node has gone through
        self.neighbors = []
        self.active = True
        self.data_size = 0
        self.last_update_round = 0
        # The model, i.e., temperature in region. 2D int array.
        self.local_data = []
        # Indicate which subregions this node sensed. This is a set of tuple(int, int). Use to make it easier to update models, rather than constantly navigating local_data in O(n^2).
        self.sensed_locations = set()
        self.message_send_success_rate = 0.85
        self.ack_send_success_rate = 0.9
        self.packets_per_message = 3

        self.voltage = 3.8 # in V
        if node_type == "CC1310":
            self.rx_current = 0.0054 # 5.4 mA
            self.tx_current = 0.0134 # 13.4 mA at 10 dBm
            # self.standby_current = 0.0000007 # 0.7 µA
        elif node_type == "CC2652R":
            self.rx_current = 0.0069 # 6.9 mA
            self.tx_current = 0.007 # 7.0 mA # at 0 dBm
            # self.standby_current = 0.00000094 # 0.94 µA
        elif node_type == "CC1352R":
            self.rx_current = 0.0058 # 5.8 mA
            self.tx_current = 0.007 # 7.0 mA # at 0 dBm
            # self.standby_current = 0.00000085 # 0.85 µA
        
    def __str__(self):
        """Helpful for debugging."""
        return f"Node {self.node_id} at {tuple(round(coord, 1) for coord in self.position)} [{round(self.energy, 2)}/{int(self.max_energy)}J], subregions sensed: {len(self.sensed_locations)}"

    def display_data(self, display_local_data=False):
        if display_local_data:
            return f"> sensed_locations ({len(self.sensed_locations)}): {self.sensed_locations}\n> local_data: {self.local_data}"
        return f"> sensed_locations ({len(self.sensed_locations)}): {self.sensed_locations}"
    
    def display_full_info(self, display_local_data=False):
        return f"{self.__str__()}\n{self.display_data(display_local_data)}"

    def consume_energy(self, amount):
        self.energy -= amount
        if self.energy <= 0:
            self.energy = 0
            self.active = False
    
    def calculate_num_messages(self, num_packets):
        """Consider each message (involving the transfer of model data) is 40 bytes, each packet (which represents the temperature of subregions) is 12 bytes, and the header (preamble, node id) is 4 bytes. Packet would contain three ints: temperature, x-coord, y-coord. Each packet represents data of the temperature of one subregion. Thus, 3 packets per message limit."""
        return ((num_packets - 1)//self.packets_per_message) + 1

    def calculate_time_to_send_messages(self, num_messages, bytes_per_message=40):
        """In order to know how long it will take to send/receive num_packets packets, we need to consider how many bytes they would require, and how long it would take to send/receive that much information over the appropriate channel.
        
        Returns: the time it would take to send/receive these messages."""
        # Assume 250 Kbps channel capacity
        # sizes of messages in bytes * 8 bits per byte / channel capacity in bps 
        return (num_messages*bytes_per_message) * 8 / 250000

    def send_model(self, target):
        """Send model to target node via messages so target can aggregate its model. Some messages may fail to send.
        
        Returns: an int representing successful messages sent, an int representing total messages sent, and two floats representing the energy consumed by both nodes."""
        
        def activity_constraint():
            print(f"Simulation Failed. A node became inactive, which is forbidden by a constraint of the simulation.")
            exit()
        
        # How big model data is
        num_packets = len(self.sensed_locations)
        num_messages = self.calculate_num_messages(num_packets)

        # NO ACKS NEEDED
        if self.resend_threshold == 0:
            # There is a chance that message(s) fails to send
            successful = [random.random() <= self.message_send_success_rate for _ in range(num_messages)]
            num_successful = sum(successful)
            num_unsuccessful = num_messages - num_successful

            # Transmitter node consumes as much energy as it sent messages
            time = self.calculate_time_to_send_messages(num_messages)
            # E = V * I * t (energy = voltage * current * time)
            self_energy_consumed = self.voltage * self.tx_current *  time
            self.consume_energy(self_energy_consumed)
            if not self.active: activity_constraint()

            # Receiver node only consumes as much energy as the number of messages it actually received
            receive_time = self.calculate_time_to_send_messages(num_messages - num_unsuccessful)
            target_energy_consumed = target.voltage * target.rx_current * receive_time
            target.consume_energy(target_energy_consumed)
            if not target.active: activity_constraint()
        # Will try to resend? Then ACKs are needed
        else:
            # There is a chance that message(s) fails to send
            # Need to consider the chance that a message will drop when sent and when ACK-ed
            sent_successful = [random.random() <= self.message_send_success_rate for _ in range(num_messages)]
            ack_successful = [random.random() <= self.ack_send_success_rate if sent_successful[i] else False for i in range(num_messages)]

            initial_sent = sum(sent_successful)
            initial_acks = sum(ack_successful)
            num_unsuccessful = num_messages - initial_acks # initial_acks ~ num_successful

            retries = 0
            additional_acks_sent = 0
            additional_acks_received = 0

            while num_unsuccessful > 0 and retries < self.resend_threshold:
                retries += 1
                # Does sender message go through?
                if random.random() <= self.message_send_success_rate:
                    additional_acks_sent += 1
                    # Does receiver ACK message go through?
                    if random.random() <= self.ack_send_success_rate:
                        additional_acks_received += 1
                        num_unsuccessful -= 1
                        ack_successful[ack_successful.index(False)] = True

            # E = V * I * t (energy = voltage * current * time)
            # SELF
            self_transmit_energy = self.voltage * self.tx_current * self.calculate_time_to_send_messages(num_messages + retries) # sending messages (local data)
            self_receive_energy = self.voltage * self.rx_current * self.calculate_time_to_send_messages(initial_acks + additional_acks_received, bytes_per_message=5) # receiving ACKs
            self_energy_consumed = self_transmit_energy + self_receive_energy
            self.consume_energy(self_energy_consumed)
            if not self.active: activity_constraint()

            # TARGET
            target_receive_energy = target.voltage * target.rx_current * self.calculate_time_to_send_messages(initial_sent + additional_acks_sent) # receive messages (data)
            target_transmit_energy = target.voltage * target.tx_current * (self.calculate_time_to_send_messages(initial_acks + additional_acks_sent, bytes_per_message=5)) # send ACKs
            target_energy_consumed = target_receive_energy + target_transmit_energy
            target.consume_energy(target_energy_consumed)
            if not target.active: activity_constraint()

            successful = ack_successful
            num_successful = sum(ack_successful)

        target.update_model(self, successful)
        return num_successful, num_messages, self_energy_consumed, target_energy_consumed
    
    def update_model(self, neighbor, successful, debug=False):
        """Aggregate your local_data model with that of the neighbor's. For subregions where both nodes have data, combine data (i.e., average).
        
        Parameters:
        -----------
        neighbor : Node
            The model this node is using to update its own model
        successful : array of bool
            Which messages successfully sent (3 temperature values per message)
        """
        i = 0
        if debug: display_info = f"Updating Model for Node {self.node_id}, successful={successful}: "
        for (x, y) in neighbor.sensed_locations:
            if not successful[i//self.packets_per_message]: # that message was not successful so skip this packet
                if debug: display_info = display_info + f"ERR({x},{y}); "
                i += 1
                continue
            if (x, y) in self.sensed_locations: # information already in model so take average
                average_value = (self.local_data[x][y] + neighbor.local_data[x][y]) / 2
                self.local_data[x][y] = average_value
                if debug: display_info = display_info + f"avg({x},{y}) = {average_value}; "
            else: # never saved such information so save directly
                self.local_data[x][y] = neighbor.local_data[x][y]
                self.sensed_locations.add((x, y))
                if debug: display_info = display_info + f"new({x},{y}) = {neighbor.local_data[x][y]}; "
            i += 1
        if debug: print(display_info + "\n")
        
        # SET MANIPULATION
        # common = self.sensed_locations & neighbor.sensed_locations
        # if debug: display_info = "N/A" if not common else ""
        # for (x, y) in common:
        #     average_value = (self.local_data[x][y] + neighbor.local_data[x][y]) / 2
        #     self.local_data[x][y] = average_value
        #     if debug: display_info = display_info + f"({x},{y}) = {average_value}; "
        
        # in_neighbor_not_self = neighbor.sensed_locations - self.sensed_locations
        # if debug:
        #     print(f"COMMON (N{self.node_id}, N{neighbor.node_id}): {display_info}")
        #     display_info = "N/A" if not in_neighbor_not_self else ""
        # for (x, y) in in_neighbor_not_self:
        #     self.local_data[x][y] = neighbor.local_data[x][y]
        #     self.sensed_locations.add((x, y))
        #     if debug: display_info = display_info + f"({x},{y}) = {neighbor.local_data[x][y]}; "
        # if debug: print(f"NEIGHBOR NOT SELF (N{self.node_id}, N{neighbor.node_id}): {display_info}\n")

    
class P2PNetwork:
    def __init__(self, n_nodes=36, area_side_length=100, area_side_segments=10, max_rounds=10, node_resend_threshold=0, uniform_locations=False, visualize=False, node_types_distribution={"CC1310":0.4, "CC2652R":0.3, "CC1352R":0.3}, debug=False):
        """
        Initialize a P2P network with heterogeneous nodes.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes in the network
        area_side_length : float
            Distance of side length in meters
        area_side_segments : int
            Number of segments the 2D matrix area_data is divided into
        max_rounds : int
            Maximum number of rounds for simulation
        node_resend_threshold : int
            Number of message resends a node is willing to send per generation
        uniform_locations : bool
            Uniform or random placement of nodes across the region
        visualize : bool
            Whether or not to display plot
        node_types_distribution : dict
            Distribution of different node types
        """
        self.n_nodes = n_nodes
        self.area_side_segments = area_side_segments
        self.area_side_length = area_side_length
        self.subregion_side_length = self.area_side_length/self.area_side_segments
        self.node_types_distribution = node_types_distribution
        self.max_rounds = max_rounds
        self.node_resend_threshold = node_resend_threshold

        self.nodes: list[Node] = []
        self.round = 0
        self.area_data = self.generate_area_data()

        connectivity = False
        attempts = 0
        max_attempts = 10

        # Constraint; network must be fully connected 
        while connectivity is False and attempts < max_attempts:
            self.create_heterogeneous_network(uniform_locations)
            self.graph = self.build_graph()
            connectivity = nx.is_connected(self.graph)
            attempts += 1

        if connectivity is False:
            print(f"Simulation Failed. Network is not fully connected. Program exceeded maximum retry allowance of {max_attempts} attempts.")
            exit()

        self.generate_local_data()

        if debug:
            print(f"Area Data ({area_side_segments} x {area_side_segments}):")
            for row in self.area_data:
                print(row)

            # Print network information 
            print(f"Number of nodes: {self.graph.number_of_nodes()}")
            print(f"Number of edges: {self.graph.number_of_edges()}")
            print()

        if visualize:
          plt = self.visualize_network(self.graph)
          plt.show()

    def set_node_resend_threshold(self, threshold):
        self.node_resend_threshold = threshold
        for node in self.nodes:
            node.resend_threshold = threshold


    def create_heterogeneous_network(self, uniform_locations=True):
        """Create the nodes of the system using node_type_distribution, which can by customized to specify different percentages of different MCU types."""
        node_types = []
        
        # Determine node types based on distribution
        for node_type, percentage in self.node_types_distribution.items():
            count = int(self.n_nodes * percentage)
            node_types.extend([node_type] * count)
        
        # If any space remaining, add CC1310
        while len(node_types) < self.n_nodes:
            node_types.append("CC1310")

        random.shuffle(node_types)

        # Create nodes
        self.nodes = []

        # Making sure number of nodes is a perfect square
        if uniform_locations:
            nodes_per_row = int(math.sqrt(self.n_nodes))
            self.n_nodes = nodes_per_row**2
            # e.g. if side length of area is 100m and there are 5 nodes per row,
            # then each node would need to be spaced at 20m intervals.
            unit_length = self.area_side_length/nodes_per_row

        for i in range(self.n_nodes):
            node_type = node_types[i]
            
            # Uniform distribution of nodes in the region
            if uniform_locations:
                pos = ((i % nodes_per_row)*unit_length + unit_length/2, int(i/nodes_per_row)*unit_length + unit_length/2)
            # Random locations in the region
            else:
                pos = (random.uniform(0, self.area_side_length), 
                        random.uniform(0, self.area_side_length))
            
            self.nodes.append(Node(
                node_id=i,
                position=pos,
                node_type=node_type,
                resend_threshold=self.node_resend_threshold
            ))

    def build_graph(self):
        """Build network graph based on node positions and transmission ranges."""
        G = nx.Graph()
         
        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(node.node_id, 
                       pos=node.position, 
                       type=node.node_type,
                       energy=node.energy,
                       max_energy=node.max_energy)
        
        # Connect nodes within transmission range
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:], i+1):
                # determine vector length
                dist = np.linalg.norm(np.array(node1.position) - np.array(node2.position))
                # Bidirectional connection if either node can reach the other
                if dist <= max(node1.transmission_range, node2.transmission_range):
                    G.add_edge(node1.node_id, node2.node_id, weight=dist)
                    node1.neighbors.append(node2.node_id)
                    node2.neighbors.append(node1.node_id)
        
        return G
    
    def visualize_network(self, G):
        """Visualize the network with nodes and connections; G is the NetworkX graph."""
        plt.figure(figsize=(10, 10))

        node_positions = {node.node_id: node.position for node in self.nodes}
        # nodes
        node_colors = {
            "CC1310": "green",
            "CC2652R": "red",
            "CC1352R": "blue"
        }
        # different MCUs -> different colors
        for node_type, color in node_colors.items():
            nodes_of_type = [node.node_id for node in self.nodes if node.node_type == node_type]
            if nodes_of_type:
              nx.draw_networkx_nodes(G, node_positions, nodelist=nodes_of_type, node_size=300, node_color=color, alpha=0.7)

        # edges
        nx.draw_networkx_edges(G, node_positions, width=1.0, alpha=0.5)

        # labels
        custom_labels = {}
        for node in self.nodes:
          custom_labels[node.node_id] = f"{node.node_id}" 
        nx.draw_networkx_labels(G, node_positions, labels=custom_labels, font_size=12, font_color="white", font_family="sans-serif")

        # transmission ranges per node
        for node in self.nodes:
          circle = plt.Circle(node.position, node.transmission_range, 
                            color="red", fill=False, linestyle="--", alpha=0.3)
          plt.gca().add_patch(circle)

        # legend
        legend_elements = []
        for node_type, color in node_colors.items():
          legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=f"= {node_type}"))
        legend = plt.legend(handles=legend_elements, loc="upper right", title="Node Types", framealpha=0.7, handletextpad=0.3)
        # legend.set_fontsize(16)
        # legend.set_fontweight("bold")

        # Set plot limits and remove axes
        plt.xlim(0, self.area_side_length)
        plt.ylim(0, self.area_side_length)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title("Network Visualization with Transmission Ranges")
        plt.tight_layout()

        return plt
    
    def generate_area_data(self, start_range=(20, 30)):
        """Create 2D array of size 'area_side_segments X area_side_segments' to represent the actual data of the area.
        
        Each element represents the temperature of a subregion of the region. Neighboring subregions will not vary by more than neighbor_variance (more or less, depends on flow of data population). Data will permeate in a DFS manner to guarantee neighboring subregions have similar temperatures.
        
        Parameters:
        -----------
        start_range : tuple(int, int)
            Starting range for the temperature value of the middlemost subregion
        """
        
        def populate_area_data(area_data, max, x, y, neighbor_value, neighbor_variance=3):
            if x < 0 or y < 0 or x >= max or y >= max or area_data[x][y]!=0:
              return
            
            val = random.randint(-neighbor_variance, neighbor_variance) + neighbor_value
            area_data[x][y] = val

            populate_area_data(area_data, max, x+1, y, neighbor_value)
            populate_area_data(area_data, max, x-1, y, neighbor_value)
            populate_area_data(area_data, max, x, y+1, neighbor_value)
            populate_area_data(area_data, max, x, y-1, neighbor_value)

        area_data = [[0 for _ in range(self.area_side_segments)] for _ in range(self.area_side_segments)]

        start_index = (self.area_side_segments//2, self.area_side_segments//2)
        populate_area_data(area_data, self.area_side_segments, start_index[0], start_index[1], random.randint(start_range[0], start_range[1]))

        return area_data

    def generate_local_data(self, sensor_error=1):
        """Generate simulated data for each node."""
        for node in self.nodes:
            local_data = [[0 for _ in range(self.area_side_segments)] for _ in range(self.area_side_segments)]
            sensed_locations = set()
            # coordinates for subregion that node is in. Node keeps track of temperature in that general area.
            node_x = int(node.position[0]/self.subregion_side_length)
            node_y = int(node.position[1]/self.subregion_side_length)
            
            # Represents how many subregion squares around the node can the node sensor detect. To be conservative, I am subtracting half the current subregion square (as if node is roughly in the middle of the subregion), then considering the maximum additional diagonal distance it can detect (in subregion squares), floored.
            furthest_detection = int((node.detection_range - self.subregion_side_length/2) / (self.subregion_side_length * math.sqrt(2)))
            for i in range(-furthest_detection, furthest_detection+1):
                for j in range(-furthest_detection, furthest_detection+1):
                    x = node_x+i
                    y = node_y+j
                    if x >=0 and y >=0 and x < self.area_side_segments and y < self.area_side_segments:
                        local_data[x][y] = self.area_data[node_x][node_y]
                        sensed_locations.add((x,y))
                        #     print(f"ERROR! Node {node}: (x,y)=({x},{y}). (i,j)=({i},{j}) (nodeX, nodeY)=({node_x},{node_y}). area_side_segments={self.area_side_segments}")
            node.local_data = local_data
            node.sensed_locations = sensed_locations

            # print(node)
            # print(f"Local Data:")
            # for row in node.local_data:
            #     print(row)
            # print()
        # print()
    
    def gossip_learning(self, neighbor_selection_strategy="random", max_neighbors_selected = 3):
        """Represents 1 round/generation of the simulation. Each node shares model with neighboring nodes."""
        self.round += 1

        reliability_data = [0 for _ in range(self.n_nodes)]
        energy_data = [0 for _ in range(self.n_nodes)]

        # Before each round, shuffle nodes to simulate random behavior
        random.shuffle(self.nodes)

        # Each node selects neighbors and shares its model so that it can be aggregated by the neighbor
        for node in self.nodes:
            # Varying neighbor selection strategies
            if neighbor_selection_strategy == "least-interacted":
                selected_neighbors = node.neighbors if len(node.neighbors) <= max_neighbors_selected else node.neighbors[0:max_neighbors_selected]
                node.neighbors = node.neighbors[max_neighbors_selected:] + node.neighbors[0:max_neighbors_selected]
            else: # "random"
                selected_neighbors = random.sample(node.neighbors, min(max_neighbors_selected, len(node.neighbors)))

            successfully_sent = 0
            total_sent = 0

            for neighbor_id in selected_neighbors:
                target = self.nodes[neighbor_id]
                num_successful, total_messages, self_energy_consumed, neighbor_energy_consumed = node.send_model(target)
                successfully_sent += num_successful
                total_sent += total_messages
                energy_data[node.node_id] += self_energy_consumed
                energy_data[neighbor_id] += neighbor_energy_consumed

            reliability_data[node.node_id] = successfully_sent / total_sent
        
        return reliability_data, energy_data
                      
    def calculate_overall_accuracy(self):
        accuracy_data = [self.calculate_accuracy(self.nodes[i]) for i in range(self.n_nodes)]
        return accuracy_data
        # most, least, total = 0.0, 100.0, 0.0
        # most_id = 0
        # for node in self.nodes:
        #     accuracy = self.calculate_accuracy(node)
        #     if accuracy > most:
        #         most = accuracy
        #         most_id = node.node_id
        #     least = min(least, accuracy)
        #     total += accuracy
        # avg = total/self.n_nodes
        
        # return avg, least, most, most_id
    
    def calculate_accuracy(self, node, debug=False):
            """
            Accuracy is measured as how close the local data is to the actual area data. Each subregion can have an accuracy from 0 to 1, with 1 being completely accurate in that subregion. Add up all the subregion accuracies and divide by the number of subregions, then multiply by 100 to get the percent representation. 
            """
            if debug:
                print(f"CALCULATING ACCURACY FOR... {node}")
            max_accuracy = self.area_side_segments**2
            accuracy = 0.0
            for (x, y) in node.sensed_locations:
                error = abs(node.local_data[x][y] - self.area_data[x][y])/self.area_data[x][y]
                accuracy += (1 - error)
                if debug: print(f"error = {round(error, 4)}, local = {node.local_data[x][y]}, area = {self.area_data[x][y]}, accuracy = {round(accuracy, 4)}/{max_accuracy}")
            accuracy = accuracy * 100 / max_accuracy
            if debug: print(f"ACCURACY = {round(accuracy, 4)}\n")
            return accuracy

    def run_simulation(self, neighbor_selection_strategy="random", max_neighbors_selected=3, visualize=False, debug=False):
        tracker = PerformanceTracker()
        for r in range(self.max_rounds):
            reliability_data, energy_data = self.gossip_learning(neighbor_selection_strategy, max_neighbors_selected)
            accuracy_data = self.calculate_overall_accuracy()
            
            tracker.update(r, reliability_data, energy_data, accuracy_data)

            if debug: 
                # self.print_nodes()
                print(f"=== Round {self.round}: avg={round(sum(accuracy_data)/len(accuracy_data),4)}, min={round(min(accuracy_data),4)}, max={round(max(accuracy_data),4)} ===")
                # print(f"> reliability_data = {reliability_data}")
                # print(f"> energy_data = {energy_data}")
                # print(f"> accuracy_data = {accuracy_data}")
            
            # End simulation if win conditions met
            if tracker.both_thresholds_reached_gen is not None:
                break
            
        if debug or visualize: print(tracker)
        if visualize:
            self.visualize_network(self.graph).show()
            tracker.plot_reliability().show()
            tracker.plot_energy_consumption().show()
            tracker.plot_accuracy().show()

        if tracker.both_thresholds_reached_gen is not None: print("\nSimulation Successful.")
        else: print("\nSimulation Failed. Did not meet requirements.")
        return tracker

    def print_nodes(self):
        for node in self.nodes:
            print(node)
  
class PerformanceTracker:
    def __init__(self):
        # Which generation are we on (helps for plotting graph)
        self.generations = []

        # Successful unique messages / Total unique messages
        self.reliability_avg = []
        self.reliability_min = []
        self.reliability_max = []

        # Energy expended
        self.energy_consumption_avg = []

        # How similar local model results are compared to area data
        self.accuracy_min = []
        self.accuracy_avg = []
        self.accuracy_max = []

        # Latency thresholds reached (in %)
        self.min_accuracy_threshold = 75
        self.avg_accuracy_threshold = 90
        self.min_threshold_reached_gen = None
        self.avg_threshold_reached_gen = None
        self.both_thresholds_reached_gen = None
    
    def __str__(self):
        return f"\nRELIABILITY:\nmin={self.reliability_min}\navg={self.reliability_avg}\nmax={self.reliability_max}\n\nENERGY:\nenergy={self.energy_consumption_avg}\n\nACCURACY:\nmin={self.accuracy_min}\navg={self.accuracy_avg}\nmax={self.accuracy_max}"

    def update(self, generation, reliability_data, energy_data, accuracy_data):
        """
        Update performance metrics with new generation data.
        
        Parameters:
        -----------
        generation : int
            Current generation/round number
        reliability_data : list
            List of reliability values for each node (successful/total msgs)
        energy_data : list
             List of energy consumption values for each node
        accuracy_data : list
            List of accuracy of that generation
        """
        self.generations.append(generation)
        
        # Update reliability (avg of successful/total messages per node)
        min_reliability = min(reliability_data)
        avg_reliability = sum(reliability_data)/len(reliability_data)
        max_reliability = max(reliability_data)
        
        self.reliability_min.append(min_reliability)
        self.reliability_avg.append(avg_reliability)
        self.reliability_max.append(max_reliability)
        
        # Update energy consumption (avg energy lost per node)
        avg_energy = sum(energy_data)/len(energy_data)
        self.energy_consumption_avg.append(avg_energy)
        
        # Update accuracy metrics
        min_accuracy = min(accuracy_data)
        avg_accuracy = sum(accuracy_data)/len(accuracy_data)
        max_accuracy = max(accuracy_data)
        
        self.accuracy_min.append(min_accuracy)
        self.accuracy_avg.append(avg_accuracy)
        self.accuracy_max.append(max_accuracy)
        
        # Check if thresholds are reached for the first time
        if min_accuracy >= self.min_accuracy_threshold and self.min_threshold_reached_gen is None:
            self.min_threshold_reached_gen = generation
            
        if (avg_accuracy >= self.avg_accuracy_threshold or avg_accuracy + 1 >= max_accuracy) and self.avg_threshold_reached_gen is None:
            self.avg_threshold_reached_gen = generation
            
        if (self.min_threshold_reached_gen is not None and 
            self.avg_threshold_reached_gen is not None and 
            self.both_thresholds_reached_gen is None):
            self.both_thresholds_reached_gen = generation

    def plot_reliability(self):
        """Plot min, avg, and max reliability over generations."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.generations, self.reliability_min, "r-", linewidth=2, label="Min Reliability")
        plt.plot(self.generations, self.reliability_avg, "b-", linewidth=2, label="Avg Reliability")
        plt.plot(self.generations, self.reliability_max, "g-", linewidth=2, label="Max Reliability")
        
        plt.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Perfect Reliability")
        
        # Set x-ticks to be only whole numbers
        whole_numbers = np.arange(int(min(self.generations)), int(max(self.generations)) + 1)
        plt.xticks(whole_numbers)

        plt.xlabel("Generation / Round")
        plt.ylabel("Reliability\n(Successful msgs / Total msgs)")
        plt.title("System Reliability Over Time")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt

    def plot_energy_consumption(self):
        """Plot energy consumption over generations."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.generations, self.energy_consumption_avg, "r-", linewidth=2, label="Avg. Energy Consumption")
        
        # Add trend line
        if len(self.generations) > 1:
            z = np.polyfit(self.generations, self.energy_consumption_avg, 1)
            p = np.poly1d(z)
            plt.plot(self.generations, p(self.generations), "r--", alpha=0.5, label="Trend")
        
        # Set x-ticks to be only whole numbers
        whole_numbers = np.arange(int(min(self.generations)), int(max(self.generations)) + 1)
        plt.xticks(whole_numbers)

        plt.xlabel("Generation / Round")
        plt.ylabel("Energy Consumption\n(Average Energy Lost)")
        plt.title("System Energy Consumption Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt

    def plot_accuracy(self):
        """Plot min, avg, and max accuracy over generations."""
        plt.figure(figsize=(10, 6))

        plt.plot(self.generations, self.accuracy_min, "r-", linewidth=2, label="Min Accuracy")
        plt.plot(self.generations, self.accuracy_avg, "b-", linewidth=2, label="Avg Accuracy")
        plt.plot(self.generations, self.accuracy_max, "g-", linewidth=2, label="Max Accuracy")

        # Add threshold lines
        plt.axhline(y=self.min_accuracy_threshold, color="r", linestyle="--", alpha=0.7, 
                   label=f"Min Threshold ({self.min_accuracy_threshold}%)")
        plt.axhline(y=self.avg_accuracy_threshold, color="b", linestyle="--", alpha=0.7,
                   label=f"Avg Threshold ({self.avg_accuracy_threshold}%)")

        # Set x-ticks to be only whole numbers
        whole_numbers = np.arange(int(min(self.generations)), int(max(self.generations)) + 1)
        plt.xticks(whole_numbers)

        if self.avg_threshold_reached_gen is not None:
            label = f"Avg ≥ {self.avg_accuracy_threshold}%"
            if self.accuracy_avg[-1] < self.avg_accuracy_threshold:
                label = "Avg + 1 ≥ Max"
            plt.axvline(x=self.avg_threshold_reached_gen, color="b", linestyle=":", alpha=0.5)
            plt.text(self.avg_threshold_reached_gen + 0.05, 50, f"{label} (Gen {self.avg_threshold_reached_gen})", rotation=90, verticalalignment="center")

        # Mark where thresholds were reached
        if self.min_threshold_reached_gen is not None:
            x_coord = self.min_threshold_reached_gen
            if self.avg_threshold_reached_gen == self.min_threshold_reached_gen:
                x_coord -= 0.02 # to avoid overlap
            plt.axvline(x=x_coord, color="r", linestyle=":", alpha=0.5)
            plt.text(self.min_threshold_reached_gen + 0.05, 15, f"Min ≥ {self.min_accuracy_threshold}% (Gen {self.min_threshold_reached_gen})", rotation=90, verticalalignment="center")
            

        if self.both_thresholds_reached_gen is not None:
            plt.axvline(x=self.both_thresholds_reached_gen + 0.02, color="g", linestyle="-", alpha=0.5,
                       label=f"Both Thresholds Met")
        
        plt.xlabel("Generation")
        plt.ylabel("Accuracy (in %)")
        plt.title("System Accuracy Over Time")
        plt.ylim(0, 100.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        
        return plt
    
    def calculate_simulation_result(self):
        """Returns the overall (reliability, energy consumption, latency) and the performance metric of the simulation. Returns None if simulation never reached win conditions."""
        reliability = sum(self.reliability_avg)/len(self.reliability_avg)
        energy_consumption = sum(self.energy_consumption_avg)/len(self.energy_consumption_avg)
        latency = self.both_thresholds_reached_gen
        if latency is None: return (reliability, energy_consumption, latency), None
        return (reliability, energy_consumption, latency), reliability / (energy_consumption * latency)

    def format_simulation_results(self, simulation_results, places=4):
        return f"(R={round(simulation_results[0], places)}, E={round(simulation_results[1],places)}, L={simulation_results[2]})"

class MultiObjOptimization:
    def __init__(self):
        self.reliability_results = []
        self.energy_consumption_results = []
        self.latency_results = []

    def update(self, simulation_result):
        """Updates information per simulation.

        Parameter simulation_result = PerformanceTracker.calculate_simulation_result()[0]"""
        self.reliability_results.append(simulation_result[0])
        self.energy_consumption_results.append(simulation_result[1])
        self.latency_results.append(simulation_result[2])

    def is_pareto_efficient(self, points):
        """
        Find the Pareto-efficient points based on optimization directions.

        Parameters:
        -----------        
        points : list of tuples
            Array of points where each row represents a point and columns represent objectives
        
        Returns: boolean mask of Pareto-efficient points
        """
        n_points, n_objectives = points.shape
        is_efficient = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if not is_efficient[i]:
                continue
                
            for j in range(n_points):
                if i != j:
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        is_efficient[i] = False
                        break
                        
        return is_efficient

    def calculate_performance_metric(self):
        """Calculate a normalized performance metric. For all metrics, higher values will represent better performance."""
        # Create copies to avoid modifying original data
        rel = np.array(self.reliability_results)
        energy = np.array(self.energy_consumption_results)
        lat = np.array(self.latency_results)
        
        # Normalize each metric to 0-1 range
        rel_norm = (rel - np.min(rel)) / (np.max(rel) - np.min(rel)) if np.max(rel) != np.min(rel) else np.ones_like(rel)
        # For energy and latency, lower is better, so invert the normalization
        energy_norm = 1 - ((energy - np.min(energy)) / (np.max(energy) - np.min(energy))) if np.max(energy) != np.min(energy) else np.ones_like(energy)
        lat_norm = 1 - ((lat - np.min(lat)) / (np.max(lat) - np.min(lat))) if np.max(lat) != np.min(lat) else np.ones_like(lat)
        
        # Weighted sum for composite score (customize weights based on importance)
        weights = [0.33, 0.33, 0.33]  # reliability, energy, latency
        return weights[0] * rel_norm + weights[1] * energy_norm + weights[2] * lat_norm


    def order_points_for_curve(self, points):
        """
        Order points to create a smooth curve along the Pareto front.
        
        Uses a simple nearest neighbor approach.
        
        Parameters:
        -----------
        points : list
            Array of points to order
        
        Returns: indices of points in order
        """
        if len(points) <= 2:
            return np.arange(len(points))
        
        # Calculate pairwise distances
        dist_matrix = squareform(pdist(points))
        
        # Start with the point that has highest reliability (index 0)
        current_idx = np.argmax(points[:, 0])
        ordered_indices = [current_idx]
        unvisited = set(range(len(points)))
        unvisited.remove(current_idx)
        
        # Greedy nearest neighbor traversal
        while unvisited:
            # Find the closest unvisited point
            current_distances = dist_matrix[current_idx]
            current_distances[list(ordered_indices)] = np.inf  # Exclude visited points
            next_idx = np.argmin(current_distances)
            
            # If we've visited all points or there's no connection, break
            if next_idx in ordered_indices or current_distances[next_idx] == np.inf:
                break
                
            ordered_indices.append(next_idx)
            unvisited.remove(next_idx)
            current_idx = next_idx
            
        return np.array(ordered_indices)

    def plot_pareto_surface(self, ax, pareto_points):
        """
        Draw a surface/mesh connecting Pareto points.
        
        Parameters:
        -----------        
        ax : Axes
            Matplotlib 3D axis
        pareto_points : list of tuples
            Array of Pareto-optimal points
        """
        # If we have few points, just connect them
        if len(pareto_points) < 4:
            # For fewer points, just draw lines between them in sorted reliability order
            sorted_indices = np.argsort(pareto_points[:, 0])
            sorted_points = pareto_points[sorted_indices]
            ax.plot(sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2], 
                    "r-", linewidth=2, alpha=0.6)
            return

        try:
            # Try to create a convex hull of points
            hull = ConvexHull(pareto_points)
            
            # Get simplices from hull
            simplices = hull.simplices
            
            # Draw each face of the hull
            for simplex in simplices:
                pts = pareto_points[simplex]
                # Draw the triangle
                ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], 
                            color="red", alpha=0.2, shade=True)
                
                # Draw the edges
                for i in range(3):
                    ax.plot([pts[i, 0], pts[(i+1)%3, 0]], 
                            [pts[i, 1], pts[(i+1)%3, 1]], 
                            [pts[i, 2], pts[(i+1)%3, 2]], 
                            "r-", linewidth=1.5, alpha=0.6)
        except Exception as e:
            # Fallback if hull creation fails: draw a curve connecting points
            # Order points for a smooth curve
            ordered_indices = self.order_points_for_curve(pareto_points)
            ordered_points = pareto_points[ordered_indices]
            
            # Draw the curve
            ax.plot(ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2], 
                    "r-", linewidth=2, alpha=0.6, label="Pareto Front Curve")

    def plot_multi_objective_3d(self, extra_visualization=False):
        """Create a 3D scatter plot for multi-objective optimization results"""
        print("\nPlotting Simulation Results.")

        # Create the 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Load your research data
        reliability, energy_consumption, latency = self.reliability_results, self.energy_consumption_results, self.latency_results

        def negate(arr):
            return [elt*-1 for elt in arr]

        # Combine objectives for Pareto front identification
        # We want to maximize reliability and minimize energy_consumption and latency
        objectives = np.column_stack([reliability, negate(energy_consumption), negate(latency)])
        pareto_mask = self.is_pareto_efficient(objectives)

        # Generally how well did this simulation do
        performance_metric = self.calculate_performance_metric()

        # Scatter plot with color gradient based on Pareto optimality
        # This is a simplified representation - you may want to implement 
        # actual Pareto front identification
        scatter = ax.scatter(reliability, energy_consumption, latency, 
                            c=performance_metric,
                            cmap="viridis", 
                            s=50, # marker size
                            alpha=0.7)
        
        # Highlight Pareto-efficient points
        pareto_scatter = ax.scatter(np.array(reliability)[pareto_mask], 
                                np.array(energy_consumption)[pareto_mask], 
                                np.array(latency)[pareto_mask],
                                color="red", 
                                s=100, 
                                label="Pareto Front",
                                edgecolors="black",
                                linewidth=1)
        
        # Draw the Pareto front curve or surface
        pareto_points = np.column_stack([
            np.array(reliability)[pareto_mask], 
            np.array(energy_consumption)[pareto_mask], 
            np.array(latency)[pareto_mask]
        ])
        
        if extra_visualization:
            self.plot_pareto_surface(ax, pareto_points)
        
        # Add projection lines to axes (only for Pareto-optimal points to reduce clutter)
        for i in range(len(reliability)):
            if pareto_mask[i]:
                # X-Y plane projection
                ax.plot([reliability[i], reliability[i]], [energy_consumption[i], energy_consumption[i]], [0, latency[i]], color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
                
                # X-Z plane projection
                ax.plot([reliability[i], reliability[i]], [0, energy_consumption[i]], [latency[i], latency[i]], color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
                
                # Y-Z plane projection
                ax.plot([0.8, reliability[i]], [energy_consumption[i], energy_consumption[i]], [latency[i], latency[i]], color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

        # Labeling
        ax.set_xlabel("Reliability", fontweight="bold")
        ax.set_ylabel("Energy Consumption", fontweight="bold")
        ax.set_zlabel("Latency", fontweight="bold")
        ax.set_title("Multi-Objective Optimization Analysis\n"
                    "P2P Federated Learning Network", fontweight="bold")
        ax.set_xlim(0.8, 1.0)
        ax.set_ylim(ymin=0)
        ax.set_zlim(zmin=0)

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, label="Composite Score")
        cbar.set_label("Composite Score (↑ better)")
        
        # Adjust the view angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
        return [i for i in range(len(self.reliability_results)) if pareto_mask[i]]

def main():
    moo = MultiObjOptimization()
    
    # Calculate reasonable area_side_segments
    area_side_length = 500
    area_side_segments = int(area_side_length/15)

    # Making the network
    network = P2PNetwork(n_nodes=250, area_side_length=area_side_length, area_side_segments=area_side_segments, max_rounds=20, node_resend_threshold=0, uniform_locations=False, visualize=True, debug=True)

    # tmp_network = copy.deepcopy(network)
    # tmp_network.run_simulation(visualize=False, debug=True)
    # exit()
    
    # print(f"Furthest detection (in squares): {round((30-network.subregion_side_length/2)/(network.subregion_side_length*math.sqrt(2)), 4)}\n")
    # network.run_simulation(neighbor_selection_strategy="least-interacted", max_neighbors_selected=1, visualize=True)

    # Multiple simulations, multiple setups
    num_sims = 0
    for random_or_not in range(2):
        for max_neighbors in range(1,6):
            for num_resends in range(0,51,5):
                for _ in range(3):
                    num_sims += 1
                    # Deep copy to obtain same initial configuration (but not change original data).
                    current_network = copy.deepcopy(network)
                    current_network.set_node_resend_threshold(num_resends)
                    tracker = current_network.run_simulation(neighbor_selection_strategy="random" if random_or_not == 0 else "least-interacted", max_neighbors_selected=max_neighbors)
                    simulation_result, performance_metric = tracker.calculate_simulation_result()
                    if performance_metric is None:
                        print(f"Simulation {tracker.format_simulation_results(simulation_result)} never obtained reached necessary conditions.")
                        continue
                    print(f"Simulation {num_sims}: {tracker.format_simulation_results(simulation_result)} -> {performance_metric:.4f}")
                    print(f"Random: {random_or_not==0}. Max Neighbors: {max_neighbors}. Resends Allowed: {num_resends}.")
                    moo.update(simulation_result)
        
    pareto_solutions = moo.plot_multi_objective_3d(extra_visualization=False)

    print("Pareto-optimal solutions:")
    for sol in pareto_solutions:
        print(f"{sol+1}. Reliability: {moo.reliability_results[sol]:.4f}, Energy: {moo.energy_consumption_results[sol]:.4f}, Latency: {moo.latency_results[sol]}")

    moo.plot_multi_objective_3d(extra_visualization=True)

    # Debugging in earlier stages of development:
    # network = P2PNetwork(4, area_side_length=75, uniform_locations=True, visualize=False)
    # nodes = network.nodes
    # a:Node = nodes[0]
    # b:Node = nodes[1]
    # print(a.display_full_info())
    # print(network.calculate_accuracy(a))
    # print(b.display_full_info())
    # print(network.calculate_accuracy(b))
    # a.send_model(b)
    # print()
    # print(a.display_full_info())
    # print(network.calculate_accuracy(a))
    # print(b.display_full_info())
    # print(network.calculate_accuracy(b))

if __name__ == "__main__":
    main()