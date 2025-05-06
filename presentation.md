---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

<style>
section {
  font-family: 'Montserrat', 'Segoe UI', sans-serif;
  padding: 40px;
  background-color: #ffffff;
  color: #333333;
}

h1 {
  background: linear-gradient(90deg, #FF5757 0%, #FF4E97 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.journey-item {
  margin-bottom: 18px;
  border-left: 4px solid #FF5757;
  padding-left: 12px;
}

.highlight {
  background: linear-gradient(90deg, #FF5757 0%, #FF4E97 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: bold;
}
</style>

# Personal Growth: The Technical Evolution

<div class="journey-item">
  <p>Python → AWS Lambda → CloudFormation → DDD/Hexagonal → LLMs</p>
  <p>Each step represents a <span class="highlight">reinvention</span> of technical knowledge</p>
  <p>From basic scripting to complex distributed systems</p>
</div>

![bg right:35% 70%](https://via.placeholder.com/500x300/FF5757/ffffff?text=)

---

# The Drive to Improve

<div class="journey-item">
  <p>Making it work is just the beginning</p>
  <p>Striving for excellence in every iteration</p>
  <p>From "good enough" to "exceptional"</p>
</div>

```python
# Example: Evolving from simple to robust
def process_data(data):
    # Version 1: Basic implementation
    return data.process()
    
    # Version 2: With error handling
    try:
        return data.process()
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
        
    # Version 3: With validation and monitoring
    @validate_input
    @monitor_performance
    def process_data(data):
        return data.process()
```

---

# Growth Through Challenge

<div class="journey-item">
  <p>Learning LLMs and agentic applications</p>
  <p>Applying DDD principles to complex domains</p>
  <p>Building scalable, maintainable architectures</p>
</div>

```python
# Example: Modern architecture with DDD
class DomainService:
    def __init__(self, repository: Repository, event_bus: EventBus):
        self.repository = repository
        self.event_bus = event_bus
    
    async def handle_command(self, command: Command):
        aggregate = await self.repository.get(command.id)
        aggregate.execute(command)
        await self.repository.save(aggregate)
        await self.event_bus.publish(aggregate.events)
```

![bg right:35% 70%](https://via.placeholder.com/500x300/FF4E97/ffffff?text=) 