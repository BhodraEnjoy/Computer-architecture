class Flight:
    def __init__(self, flight_number, airline, destination, departure_time):
        self.flight_number = flight_number
        self.airline = airline
        self.destination = destination
        self.departure_time = departure_time
        self.passengers = []
        self.baggage = {}

    def add_passenger(self, passenger):
        self.passengers.append(passenger)

    def add_baggage(self, passenger_name, baggage_weight):
        if passenger_name not in self.baggage:
            self.baggage[passenger_name] = baggage_weight
        else:
            self.baggage[passenger_name] += baggage_weight

    def display_details(self):
        print(f"Flight {self.flight_number} by {self.airline} to {self.destination} departing at {self.departure_time}")
        print("Passengers:")
        for passenger in self.passengers:
            print(f"- {passenger}")
        print("Baggage:")
        for passenger, weight in self.baggage.items():
            print(f"- {passenger}: {weight}kg")

class Airport:
    def __init__(self, name):
        self.name = name
        self.flights = []

    def add_flight(self, flight):
        self.flights.append(flight)

    def display_flights(self):
        print(f"Flights at {self.name}:")
        for flight in self.flights:
            flight.display_details()

class FederatedAirportSystem:
    def __init__(self):
        self.airports = {}

    def add_airport(self, airport):
        self.airports[airport.name] = airport

    def display_all_flights(self):
        print("All flights in the federated airport system:")
        for airport_name, airport in self.airports.items():
            airport.display_flights()

# Example usage:
if __name__ == "__main__":
    airport1 = Airport("Airport 1")
    flight1 = Flight("ABC123", "Airline A", "New York", "10:00")
    flight1.add_passenger("John Smith")
    flight1.add_baggage("John Smith", 20)
    flight1.add_passenger("Alice Johnson")
    flight1.add_baggage("Alice Johnson", 15)
    airport1.add_flight(flight1)

    airport2 = Airport("Airport 2")
    flight2 = Flight("DEF789", "Airline B", "Los Angeles", "12:00")
    flight2.add_passenger("Bob Brown")
    flight2.add_baggage("Bob Brown", 18)
    flight2.add_passenger("Eve Green")
    flight2.add_baggage("Eve Green", 22)
    airport2.add_flight(flight2)

    federated_system = FederatedAirportSystem()
    federated_system.add_airport(airport1)
    federated_system.add_airport(airport2)

    federated_system.display_all_flights()
