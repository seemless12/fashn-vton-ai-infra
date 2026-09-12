import requests
import time
import os

API_URL = "http://52.66.205.255:8000/api/try-on"
POLL_URL = "http://52.66.205.255:8000/api/try-on/"

def test_inference(test_name, steps=15):
    print(f"\n--- Starting {test_name} ---")
    
    # We need dummy images. Let's create two small solid-color images.
    from PIL import Image
    import io
    
    img1 = Image.new('RGB', (100, 100), color = 'red')
    img2 = Image.new('RGB', (100, 100), color = 'blue')
    
    buf1 = io.BytesIO()
    img1.save(buf1, format='JPEG')
    buf1.seek(0)
    
    buf2 = io.BytesIO()
    img2.save(buf2, format='JPEG')
    buf2.seek(0)
    
    files = {
        'person_image': ('person.jpg', buf1, 'image/jpeg'),
        'garment_image': ('garment.jpg', buf2, 'image/jpeg')
    }
    
    data = {
        'steps': steps,
        'guidance_scale': 1.5
    }
    
    print("Submitting request...")
    start_time = time.time()
    res = requests.post(API_URL, files=files, data=data)
    
    if res.status_code != 200:
        print(f"Error submitting: {res.text}")
        return
        
    job = res.json()
    job_id = job['job_id']
    poll_endpoint = job['poll_endpoint']
    print(f"Job started: {job_id}")
    
    # Poll
    while True:
        poll_res = requests.get(POLL_URL + poll_endpoint)
        if poll_res.status_code != 200:
            print(f"Poll error: {poll_res.text}")
            return
            
        status = poll_res.json()
        if status['status'] == 'completed':
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Success! Total time: {elapsed:.2f}s, Generation Time: {status.get('generation_time', 'N/A')}s, Steps: {status.get('steps', 'N/A')}")
            break
        elif status['status'] == 'failed':
            print(f"Failed: {status.get('error')}")
            break
            
        print(f"Status: {status['status']}... ({time.time()-start_time:.1f}s)")
        time.sleep(2)

if __name__ == "__main__":
    test_inference("Warmup Run (Compilation)", steps=15)
    for steps in [10, 15, 25, 30]:
        test_inference(f"Benchmark {steps} Steps", steps=steps)
