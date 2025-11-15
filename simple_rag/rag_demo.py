import sys
print("using py executable: ", sys.executable, end="\n\n")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 1) prepare docs (list of strings)
docs = [
    "Legacy system stores customer names and sensitive personal ID in plaintext.",
    "We plan to migrate from monolith to microservices using APIs and a new data model.",
    "Legacy system was built in 2005 using Java 6 and Oracle 10g database with stored procedures for core logic.",
    "The modernization approach recommends migrating from monolithic to microservice-based architecture.",
    "Sensitive data such as PAN numbers and email IDs must be masked before migrating to the cloud.",
    "Authentication will move from basic auth to OAuth 2.0 with JWT-based tokens.",
    "All PII fields should be encrypted using AES-256 encryption before storage.",
    "Data migration scripts must ensure referential integrity across user and transaction tables.",
    "Existing reports will be rebuilt using Power BI instead of Crystal Reports.",
    "Legacy batch jobs running on cron will be containerized and scheduled using Kubernetes CronJobs.",
    "The front-end will shift from JSP pages to a React-based UI consuming REST APIs.",
    "All inter-service communication must use REST over HTTPS with TLS 1.3 enabled.",
    "PII data fields such as address and phone number will be tokenized during data transfer.",
    "A new audit logging mechanism will record all CRUD operations in MongoDB collections.",
    "The API gateway will handle authentication, rate limiting, and request tracing.",
    "Legacy stored procedures performing business logic will be refactored into Python microservices.",
    "System performance metrics will be captured using Prometheus and visualized in Grafana dashboards.",
    "User credentials will no longer be stored locally; they will be managed through a centralized identity provider.",
    "Database migration involves converting schema definitions to PostgreSQL-compatible syntax.",
    "Legacy FTP-based file exchange will be replaced with secure S3 bucket uploads via API.",
    "Deployment pipelines will be automated using GitHub Actions and Helm charts.",
    "Backward compatibility for old SOAP clients will be maintained for the first 6 months.",
    "Error handling and exception logs will follow structured JSON format for centralized monitoring.",
    "A dedicated module will sanitize inputs to prevent SQL injection and XSS attacks.",
    "Data validation rules from legacy code will be externalized into a validation service.",
    "All internal services will communicate using service discovery through Consul.",
    "Archival data older than 7 years will be stored in cold storage with restricted access.",
    "Legacy system used hardcoded credentials; modernization removes all hardcoded secrets.",
    "CI/CD pipelines will include automated unit and integration tests using PyTest and Postman collections.",
    "The modernization roadmap includes three phases: data cleanup, API development, and UI rebuild.",
    "Logging and monitoring will comply with SOC 2 and ISO 27001 standards.",
    "Data migration testing will include random sampling of 5% of records for validation.",
    "Sensitive PDFs and documents will be watermarked and stored in an encrypted S3 bucket.",
    "Each microservice will maintain its own database schema to avoid coupling.",
    "Redis caching layer will be introduced to reduce DB query load for frequently accessed data.",
    "Legacy XML message format will be converted to JSON during API transformation.",
    "The modernization project includes building a knowledge graph of entity relationships for better search.",
    "All services will follow OpenAPI 3.0 specifications for documentation.",
    "Production deployments will require dual approvals and blue-green strategy.",
    "Data lineage will be tracked during migration to maintain traceability.",
    "Legacy VB6 desktop clients will be replaced by web-based dashboards.",
    "Error messages will no longer expose internal stack traces to end users.",
    "The modernization plan includes integrating AI-based log anomaly detection.",
    "Data backup policy mandates daily incremental and weekly full backups.",
    "A rollback strategy must be prepared for every release to minimize downtime.",
    "All endpoints must return standardized response structures with status codes.",
    "Legacy COBOL components will be wrapped using Python REST APIs for transitional support.",
    "Each API call should include a unique correlation ID for distributed tracing.",
    "Developers must sanitize logs to ensure no PII information is stored in plaintext.",
    "Environment-specific configurations will be managed via environment variables, not code constants.",
    "All API payloads exceeding 5MB must be handled through async file upload endpoints.",
    "Sensitive configuration files will be stored in a secure vault such as HashiCorp Vault.",
    "System modernization should reduce average response time by 40% while maintaining 99.9% uptime."
]

# 2) split & embed
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = []
for d in docs:
    chunks += splitter.split_text(d)

# 3) embeddings + vector DB (Chroma local)
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(chunks, emb)

# 4) LLM
llm = ChatOllama(
    model="phi3",
    temperature=0.7,
    disable_streaming=False,
    callbacks=[StreamingStdOutCallbackHandler()],)

# 5) retrieval QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

query = "How will we handle PII during migration?"
print("\n#########################################################")
print("Query: ", query, end="\n\nResponse:\n")

output = qa.invoke({'query': query})

