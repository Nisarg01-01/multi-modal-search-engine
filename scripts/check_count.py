import weaviate

try:
    print("Connecting to Weaviate...", flush=True)
    client = weaviate.connect_to_local()
    
    if client.collections.exists("Product"):
        collection = client.collections.get("Product")
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count
        print(f"Total Products in Weaviate: {count}")
    else:
        print("Collection 'Product' does not exist.")
        
    client.close()
except Exception as e:
    print(f"Error checking count: {e}")
