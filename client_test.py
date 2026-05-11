# client_test.py
import requests
import threading

SERVER_URL = "http://127.0.0.1:8000/generate"


def send_request(user_id):
    # This is the payload being sent over the network
    payload = {
        "query": "Explain load balancing strategies in GPU clusters.",
        "user_id": user_id
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        data = response.json()
        print(
            f"User-{user_id:02d} | Worker: {data['worker_id']} | Latency: {data['latency_ms']:.1f}ms | Status: {data['status']}")
    except Exception as e:
        print(f"User-{user_id:02d} | Request Failed: {e}")


def run_networked_load_test(num_users=20):
    print(f"Simulating {num_users} concurrent external API requests...")
    threads = []

    for uid in range(num_users):
        t = threading.Thread(target=send_request, args=(uid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("All external API requests completed.")


if __name__ == "__main__":
    # Feel free to change the 20 to a higher number to stress test it!
    run_networked_load_test(20)