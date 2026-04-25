# CSE354 Distributed System Project — Team Handover (Short Version)

## Current Status

The system is **working** and already meets most requirements from the project specification.
Core functionality is implemented and tested.

What is DONE:

* Round Robin load balancing
* Least Connections load balancing
* Load-aware routing
* Multiple worker processes
* Task distribution across workers
* LLM request simulation
* RAG knowledge retrieval
* Fault tolerance (failure detection + restart)
* Retry mechanism
* Metrics collection (latency, throughput, success rate)
* 1000 concurrent users simulation

System runs successfully using:

python main.py

---

# What Is LEFT (Very Important)

These are the only things blocking us from the highest grade.

## 1) Scalability Sweep (Required)

Run the system with:

100 users
250 users
500 users
750 users
1000 users

Record:

* throughput
* latency (p95)
* success rate

Save results for the report.

Estimated time:

20–30 minutes

---

## 2) Worker Scaling Test (Required)

Run same workload (example: 500 users) using:

1 worker
2 workers
4 workers
8 workers

Goal:

Show performance improves with more workers.

Estimated time:

15 minutes

---

## 3) Save Sample Output

Create file:

sample_output.txt

Must contain:

* load testing logs
* failure simulation logs
* metrics summary

---

## 4) Write Final Report (MOST IMPORTANT)

Without the report, we cannot get a high grade.

Report must include:

* System architecture
* Load balancing explanation
* Fault tolerance explanation
* Testing results
* Scalability results
* Limitations
* References

---

## 5) Add References

Add at least:

3–5 sources

Examples:

* RAG paper
* Load balancing algorithms
* Distributed systems textbook
* Python multiprocessing documentation

Time required:

10 minutes

---

## 6) Record Demo Video (Required)

We must submit:

YouTube demo

Show:

1. Start system
2. Run load test
3. Kill worker
4. Show recovery
5. Show metrics

Length:

5–8 minutes

---

# How To Run The System

Install Python 3.9+

Then run:

python main.py

---

# Known Limitations (Normal)

* Workers are simulated (not real GPUs)
* LLM is simulated
* RAG uses simple keyword matching
* Runs on one machine

These are acceptable.

---

# Suggested Task Split

Person 1:

Run scalability sweep

Person 2:

Run worker scaling test

Person 3:

Write report

Person 4:

Record demo video

---

# Bottom Line

Code:

DONE

System:

WORKING

Remaining work:

Mostly testing evidence and documentation

We are close to the final submission.
