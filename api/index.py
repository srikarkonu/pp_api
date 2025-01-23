from flask import Flask, request, jsonify
import pandas as pd
from geopy.distance import geodesic

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Datasampling-konu - Sheet1.csv')

# Convert price from INR to Crore for better readability
def price_conversion(amount):

    def convert_to_crore(amount):
        return round(amount / 10000000, 2)

    # Ensure the conversion is correct or skip the conversion entirely.
    min_crore = convert_to_crore(amount[0]) if amount[0] else amount[0]
    max_crore = convert_to_crore(amount[1]) if amount[1] else amount[1]
    return [min_crore, max_crore]


class Rules:
    def __init__(self) -> None:
        self.df = df
        self.min_price = None
        self.max_price = None
        self.area = None

    def select_city(self, city):
        if isinstance(city, str):
            self.df = self.df[self.df['city'].str.lower() == city.lower()]
        else:
            raise ValueError("City must be a string")

    def select_locality(self, locality):
        if isinstance(locality, str):
            self.df = self.df[self.df['address'].str.lower() == locality.lower()]
        else:
            raise ValueError("Locality must be a string")

    def pincode_filter(self, pincode):
        if isinstance(pincode, int):
            self.df = self.df[self.df['pincode'] == pincode]
        else:
            raise ValueError("Pincode must be an integer")

    def expected_min_max(self):
        # Convert 'new price' to numeric, coercing errors
        self.df['new price'] = pd.to_numeric(self.df['new price'], errors='coerce')
        min_max = self.df['new price'].dropna()
        self.min_price = min_max.min() if not min_max.empty else 0
        self.max_price = min_max.max() if not min_max.empty else 0

    def bedroom_price(self, rooms=1):
        if isinstance(rooms, int) and rooms > 0:
            self.df = self.df[self.df['BHK'].fillna(0).astype(int) == rooms]
        else:
            raise ValueError("Rooms must be a positive integer")

    def area_price(self, area):
        if isinstance(area, (int, float)) and area > 0:
            # Store the total area value
            self.area = area

            # Validate that min_price and max_price are numeric
            if not isinstance(self.min_price, (int, float)):
                self.min_price = float(self.min_price) if self.min_price else 0.0
            if not isinstance(self.max_price, (int, float)):
                self.max_price = float(self.max_price) if self.max_price else 0.0

            # Adjust min_price and max_price based on area
            self.min_price *= area
            self.max_price *= area

            # Ensure that min_price is always less than max_price
            if self.min_price > self.max_price:
                self.min_price, self.max_price = self.max_price, self.min_price
        else:
            raise ValueError("Area must be a positive number")

    def gated_community_price(self):
        self.df = self.df[self.df['type'] == 'Gated community']

    def stand_alone_apartments(self):
        self.df = self.df[self.df['type'] == 'Stand Alone']

    def commercial_Spaces(self):
        self.df = self.df[self.df['type'] == 'Commercial Spaces']

    
    def filter_by_lat_long(self, lat, long, radius_km=2):
    # Ensure 'price' column exists in the DataFrame
        if 'new price' not in self.df.columns:
            raise ValueError("'new price' column not found in the properties data")

    # Drop rows with NaN latitude/longitude
        self.df = self.df.dropna(subset=['latitude', 'longitude'])

        if isinstance(lat, (int, float)) and isinstance(long, (int, float)):
            def is_within_range(row):
                try:
                    property_coords = (float(row['latitude']), float(row['longitude']))
                    user_coords = (lat, long)
                    distance = geodesic(user_coords, property_coords).km
                    return distance <= radius_km
                except Exception as e:
                    print(f"Error processing row: {row}, Error: {e}")
                    return False

        # Filter the properties by applying the distance check
            self.df = self.df[self.df.apply(is_within_range, axis=1)]
        else:
            raise ValueError("Latitude and Longitude must be numeric")

    def return_df(self):
        return self.df



@app.route('/filter_properties/', methods=['POST'])
def filter_properties():
    try:
        # Get filters from the request body
        filters = request.get_json()
        if not filters:
            return jsonify({"error": "Invalid or missing data in request body"}), 400

        rules = Rules()

        # Apply filters
        if 'city' in filters:
            rules.select_city(filters['city'])
        if 'locality' in filters:
            rules.select_locality(filters['locality'])
        if 'pincode' in filters:
            try:
                rules.pincode_filter(int(filters['pincode']))  # Ensure pincode is integer
            except ValueError:
                return jsonify({"error": "Pincode must be an integer"}), 400
        if 'property_category' in filters:
            if filters['property_category'] == 'Gated community':
                rules.gated_community_price()
            elif filters['property_category'] == 'Stand Alone':
                rules.stand_alone_apartments()
            elif filters['property_category'] == 'Commercial':
                rules.commercial_Spaces()
        if 'bedrooms' in filters:
            try:
                rules.bedroom_price(int(filters['bedrooms']))  # Ensure bedrooms is an integer
            except ValueError:
                return jsonify({"error": "Bedrooms must be a positive integer"}), 400

        if 'area' in filters:
            try:
                area = float(filters['area'])
                if area <= 0:
                    raise ValueError("Area must be greater than 0.")
                # Calculate min and max prices before area adjustment
                rules.expected_min_max()
                rules.area_price(area)
            except ValueError as ve:
                return jsonify({"error": "Invalid area value", "message": str(ve)}), 400
        else:
            # Calculate min and max prices without area adjustment
            rules.expected_min_max()
            rules.area_price(1.0)

        # Get the filtered dataframe
        filtered_properties = rules.return_df()

        if filtered_properties.empty:
            return jsonify({
                "message": "No properties found matching the given filters",
                "min_price": rules.min_price,
                "max_price": rules.max_price
            }), 404

        # Pagination logic
        page = filters.get('page', 1)
        page_size = filters.get('page_size', 10)
        start = (page - 1) * page_size
        end = start + page_size
        current_properties_page = filtered_properties[start:end]

        # Convert DataFrame to JSON
        properties_list = current_properties_page.to_dict(orient="records")

        # Return the filtered properties along with min and max prices
        return jsonify({
            "properties": properties_list,
            "min_price": rules.min_price,
            "max_price": rules.max_price
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@app.route('/available_properties', methods=['GET'])
def available_properties():
    try:
        location = request.args.get('location')

        if not location:
            return jsonify({"error": "Location parameter is required"}), 400

        # Convert location and address column to lowercase for case-insensitive filtering
        df['address'] = df['address'].str.lower()
        location = location.lower()

        # Filter properties by location (case-insensitive)
        filtered_df = df[df['address'] == location]

        if filtered_df.empty:
            return jsonify({"message": "No properties available in this location"}), 404

        return jsonify(filtered_df.to_dict(orient='records'))

    except Exception as e:
        print(f"Error occurred in available_properties endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@app.route('/budget_properties', methods=['GET'])
def budget_properties():
    try:
        # Get the query parameters
        locality = request.args.get('locality')  # Changed 'area' to 'locality'
        budget = request.args.get('budget')

        if not locality or not budget:
            return jsonify({"error": "Locality and budget parameters are required"}), 400

        try:
            budget = float(budget)  # Convert budget to float
        except ValueError:
            return jsonify({"error": "Invalid budget value. Budget must be a number"}), 400

        # Convert 'new price' column to numeric (if not already done)
        df['new price'] = pd.to_numeric(df['new price'], errors='coerce')

        # Filter properties by locality and budget (case-insensitive)
        locality = locality.lower()  # Convert locality to lowercase for case-insensitive filtering
        filtered_df = df[(df['address'].str.lower() == locality) & (df['new price'] <= budget)]

        if filtered_df.empty:
            return jsonify({"message": "No properties found under the specified budget in this locality"}), 404

        return jsonify(filtered_df.to_dict(orient='records'))

    except Exception as e:
        print(f"Error in budget_properties endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


@app.route('/market_value', methods=['GET'])
def market_value():
    location = request.args.get('location')
    property_category = request.args.get('property_category')

    if not location:
        return jsonify({"error": "Location parameter is required"}), 400

    # Normalize location to lowercase to make it case-insensitive
    location = location.strip().lower()

    # Filter by location (case-insensitive)
    filtered_df = df[df['address'].str.strip().str.lower() == location]

    # Print for debugging
    print(f"Filtering by location: {location}")

    # Further filter by property category if provided
    if property_category:
        # Strip and convert property category to lowercase for comparison
        property_category = property_category.strip().lower()
        print(f"Filtering by property category: {property_category}")
        filtered_df = filtered_df[filtered_df['type'].str.strip().str.lower() == property_category]

    if filtered_df.empty:
        return jsonify({"message": "No data available for the specified location and property category"}), 404

    # Convert 'new price' to numeric to avoid issues
    filtered_df['new price'] = pd.to_numeric(filtered_df['new price'], errors='coerce')

    # Calculate statistics
    min_price = filtered_df['new price'].min()
    max_price = filtered_df['new price'].max()
    mean_price = filtered_df['new price'].mean()
    median_price = filtered_df['new price'].median()
    mode_price = filtered_df['new price'].mode().iloc[0] if not filtered_df['new price'].mode().empty else None

    return jsonify({
        "location": location,
        "property_category": property_category or "All Categories",
        "min_price": int(min_price),
        "max_price": int(max_price),
        "mean_price": float(mean_price),
        "median_price": float(median_price),
        "mode_price": float(mode_price) if mode_price else None,
    })

@app.route('/properties_near', methods=['GET'])
def properties_near():
    try:
        # Get latitude, longitude, and radius from the query parameters
        lat = request.args.get('latitude')
        long = request.args.get('longitude')
        radius = request.args.get('radius', 2)  # Default radius to 2 if not provided

        # Validate that latitude, longitude, and radius are provided and numeric
        try:
            lat = float(lat)
            long = float(long)
            radius_km = float(radius)
        except (TypeError, ValueError):
            return jsonify({"error": "Latitude, longitude, and radius must be valid numbers"}), 400

        # Check if latitude and longitude are within valid ranges
        if not (-90 <= lat <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400
        if not (-180 <= long <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400
        if radius_km <= 0:
            return jsonify({"error": "Radius must be a positive number"}), 400

        # Debugging: Print the received values
        print(f"Received parameters: latitude={lat}, longitude={long}, radius={radius_km}")

        # Filter the properties by their distance from the given coordinates
        rules = Rules()
        rules.filter_by_lat_long(lat, long, radius_km)
        filtered_properties = rules.return_df()

        # Debugging: Print the columns of the DataFrame
        print(f"Columns in filtered_properties: {filtered_properties.columns}")

        # Ensure 'new price' column exists
        if 'new price' not in filtered_properties.columns:
            return jsonify({"error": "'new price' column not found in the properties data"}), 400

        # Convert 'new price' to numeric, invalid values will be NaN
        filtered_properties['new price'] = pd.to_numeric(filtered_properties['new price'], errors='coerce')

        # Drop rows where 'new price' is NaN
        filtered_properties = filtered_properties.dropna(subset=['new price'])

        # Calculate the price statistics (min, max, mean, median, mode)
        price_stats = {
            "min_price": float(filtered_properties['new price'].min()),  # Convert to float
            "max_price": float(filtered_properties['new price'].max()),  # Convert to float
            "mean_price": float(filtered_properties['new price'].mean()),  # Convert to float
            "median_price": float(filtered_properties['new price'].median()),  # Convert to float
            "mode_price": float(filtered_properties['new price'].mode())  # Mode 
        }

        # Return the statistics along with the filtered properties
        return jsonify({
            "min_price": price_stats["min_price"],
            "max_price": price_stats["max_price"],
            "mean_price": price_stats["mean_price"],
            "median_price": price_stats["median_price"],
            "mode_price": price_stats["mode_price"],
            "properties": filtered_properties.to_dict(orient='records')
        })

    except Exception as e:
        # Catch all errors and print the exception for debugging
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/properties_near_metro_station', methods=['GET'])
def properties_near_metro_station():
    try:
        station_name = request.args.get('station_name')
        radius = request.args.get('radius', 2)  # Default to 2 km if not provided

        if not station_name:
            return jsonify({"error": "Metro station name is required"}), 400

        # Hardcoded metro stations for demonstration
        metro_stations = {
            "Ameerpet": (17.434803, 78.448011),
            "Assembly": (17.3978004, 78.4699168),
            "Balanagar": (17.4965811, 78.3676732),
            "Begumpet": (17.4375, 78.456667),
            "Bharat Nagar": (17.463997, 78.4278693),
            "Chaitanyapuri": (17.3682882, 78.5357829),
            "Chikkadpally": (17.40036, 78.4949),
            "Dilsukhnagar": (17.3686, 78.5257),
            "Durgam Cheruvu": (17.442778, 78.3875),
            "Erragadda": (17.4567784, 78.4304257),
            "ESI Hospital": (17.4474, 78.43835),
            "Gandhi Bhavan": (17.348426, 78.550959),
            "Gandhi Hospital": (17.42552, 78.50196),
            "Habsiguda": (17.4337, 78.5016),
            "Hitec City": (17.448889, 78.383056),
            "Irrum Manzil": (17.4204695, 78.4539726),
            "JBS Parade Ground": (17.436793, 78.443906),
            "JNTU College": (17.498653, 78.388793),
            "Jubilee Hills Check Post": (17.416471, 78.438247),
            "Road No.5 Jubilee Hills": (17.43005, 78.4232),
            "Khairatabad": (17.41275, 78.45803),
            "KPHB Colony": (17.493780, 78.401795),
            "Kukatpally": (17.485116, 78.409369),
            "LB Nagar": (17.348426, 78.550959),
            "Lakdi-ka-pul": (17.404701, 78.464294),
            "Madhapur": (17.4372, 78.3982),
            "Madhura Nagar": (17.4225704, 78.37887),
            "Malakpet": (17.3772, 78.494),
            "Mettuguda": (17.4355, 78.5196),
            "MG Bus Station": (17.378055, 78.480005),
            "Miyapur": (17.4964, 78.3731),
            "Moosapet": (17.473961, 78.42044),
            "Musarambagh": (17.3711, 78.512),
            "Musheerabad": (17.425544, 78.503795),
            "Nagole": (17.3908477, 78.5587195),
            "Nampally": (17.4367, 78.4674),
            "Narayanaguda": (17.39436, 78.48996),
            "New Market": (17.3734, 78.5031),
            "NGRI": (17.41483, 78.54634),
            "Osmania Medical College": (17.382389, 78.478957),
            "Parade Grounds": (	17.436793, 78.443906),
            "Paradise": (17.4435274, 78.4850961),
            "Peddamma Gudi": (17.43065, 78.40837),
            "Prakash Nagar": (17.4425252, 78.4704161),
            "Punjagutta": (17.436793, 78.443906),
            "RTC X Roads": (17.406599, 78.496959),
            "Raidurg": (17.4422, 78.3773),
            "Rasoolpura": (17.443333, 78.475833),
            "S.R Nagar": (17.440269, 78.441833),
            "Secunderabad West": (17.4338, 78.49954),
            "Secunderabad East": (17.4337, 78.5016),
            "Stadium": (17.398795, 78.553844),
            "Sultan Bazaar": (17.385983, 78.480344),
            "Tarnaka": (17.4279, 78.536),
            "Uppal": (17.3987948, 78.5538439),
            "Victoria Memorial": (17.348426, 78.550959),
            "Yusufguda": (17.435083, 78.426528)
        }

        # Normalize station_name to title case
        station_name = station_name.strip().title()

        station_coords = metro_stations.get(station_name)

        if not station_coords:
            return jsonify({"error": "Invalid or unknown metro station name"}), 400

        lat, long = station_coords
        radius_km = float(radius)

        rules = Rules()
        rules.filter_by_lat_long(lat, long, radius_km)

        filtered_properties = rules.return_df()

        if filtered_properties.empty:
            return jsonify({"message": "No properties found near the specified metro station"}), 404

        return jsonify(filtered_properties.to_dict(orient='records'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    

@app.route('/properties_near_it_hub', methods=['GET'])
def properties_near_it_hub():
    try:
        hub_name = request.args.get('hub_name')
        radius = request.args.get('radius', 2)  # Default to 2 km if not provided

        if not hub_name:
            return jsonify({"error": "Hub name is required"}), 400

        # Hardcoded IT hubs for demonstration
        it_hubs = {
            "Hitec City": (17.44155, 78.38264),
            "Madhapur": (17.448294, 78.391487),
            "Gachibowli": (17.440081, 78.348915),
            "Kondapur": (17.467579, 78.692345),
            "Financial District": (17.4117312, 78.3424898),
            "Nanakramguda": (17.4117312, 78.3424898),
            "Manikonda": (17.4000018, 78.3861896794107),
            "Raidurg": ( 17.416315, 78.389847),
            "Uppal": (17.401810, 78.560188),
            "Pocharam": (17.173595, 78.607178)
        }

        # Normalize hub_name to title case
        hub_name = hub_name.strip().title()

        hub_coords = it_hubs.get(hub_name)

        if not hub_coords:
            return jsonify({"error": "Invalid or unknown IT hub name"}), 400

        lat, long = hub_coords
        radius_km = float(radius)

        rules = Rules()
        rules.filter_by_lat_long(lat, long, radius_km)

        filtered_properties = rules.return_df()

        if filtered_properties.empty:
            return jsonify({"message": "No properties found near the specified IT hub"}), 404

        return jsonify(filtered_properties.to_dict(orient='records'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

    
@app.route('/rera_approved', methods=['GET'])
def rera_approved_properties():
    try:
        # Retrieve project_name and location from query parameters
        project_name = request.args.get('project_name', "").strip().lower()
        location = request.args.get('location', "").strip().lower()

        # Check required columns
        required_columns = ['RERA Approved', 'Project', 'address']
        for column in required_columns:
            if column not in df.columns:
                return jsonify({
                    "error": f"'{column}' column not found in the dataset",
                    "available_columns": list(df.columns)
                }), 400

        # Filter for RERA-approved properties
        rera_approved_df = df[df['RERA Approved'].str.strip().str.lower() == 'yes']

        # If both project_name and location are not provided, return all RERA-approved properties
        if project_name == "" and location == "":
            properties = rera_approved_df.to_dict(orient='records')
            return jsonify({"rera_approved_properties": properties}), 200

        # Apply project name filter if provided
        if project_name:
            rera_approved_df = rera_approved_df[
                rera_approved_df['Project'].str.strip().str.lower().str.contains(project_name, na=False)
            ]

        # Apply location filter if provided
        if location:
            rera_approved_df = rera_approved_df[
                rera_approved_df['address'].str.strip().str.lower().str.contains(location, na=False)
            ]

        # If no properties found after applying filters, return message
        if rera_approved_df.empty:
            return jsonify({"message": "No RERA-approved properties found matching the criteria"}), 404

        # Convert to JSON and return
        properties = rera_approved_df.to_dict(orient='records')
        return jsonify({"rera_approved_properties": properties}), 200

    except Exception as e:
        print(f"Error in rera_approved_properties endpoint: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/calculate_emi', methods=['POST'])
def get_emi():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if all required keys are present in the input
        required_keys = ['loan_amount', 'tenure_years', 'annual_interest_rate']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing input: {', '.join(missing_keys)}"}), 400

        # Extract loan parameters from the request data
        try:
            loan_amount = float(data['loan_amount'])
            tenure_years = float(data['tenure_years'])
            annual_interest_rate = float(data['annual_interest_rate'])
        except ValueError:
            return jsonify({"error": "Invalid input types. All inputs must be numeric."}), 400

        # Validate input values
        if loan_amount <= 0 or tenure_years <= 0 or annual_interest_rate <= 0:
            return jsonify({"error": "All input values must be positive numbers."}), 400

        # Calculate EMI and total interest
        emi, total_interest = calculate_emi(loan_amount, annual_interest_rate, tenure_years)

        # Return result as JSON response
        return jsonify({
            "loan_amount": loan_amount,
            "annual_interest_rate": annual_interest_rate,
            "tenure_years": tenure_years,
            "emi": round(emi, 2),
            "total_interest": round(total_interest, 2)
        }), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500


def calculate_emi(loan_amount, annual_interest_rate, tenure_years):
    # Monthly interest rate
    monthly_interest_rate = (annual_interest_rate / 100) / 12

    # Total number of monthly payments
    total_months = tenure_years * 12

    # EMI formula
    emi = (loan_amount * monthly_interest_rate * ((1 + monthly_interest_rate) ** total_months)) / (((1 + monthly_interest_rate) ** total_months) - 1)

    # Total payment over the tenure
    total_payment = emi * total_months

    # Total interest
    total_interest = total_payment - loan_amount

    return emi, total_interest

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

