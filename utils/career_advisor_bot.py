# Conversational AI Career Advisor
# This module implements a rule-based chatbot with NLP capabilities for career advising

import re
import random
import string
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK data download failed. Some features may not work properly.")

class CareerAdvisorBot:
    """
    Conversational AI for providing career advice and course recommendations.
    Implements a hybrid approach combining rule-based responses with NLP for intent detection.
    """
    
    def __init__(self, knowledge_base_path=None):
        """
        Initialize the career advisor bot.
        
        Parameters:
        -----------
        knowledge_base_path : str, optional
            Path to a JSON file containing the knowledge base.
        """
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize conversation state
        self.conversation_history = []
        self.current_user = None
        self.user_profile = {}
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Prepare intents for classification
        self.intents = self.knowledge_base.get("intents", [])
        self.courses = self.knowledge_base.get("courses", [])
        self.career_paths = self.knowledge_base.get("career_paths", [])
        
        # Initialize vectorizer for intent classification
        self.vectorizer = TfidfVectorizer()
        
        # Train the vectorizer if we have intents
        if self.intents:
            # Create a corpus of all intent patterns
            corpus = []
            self.intent_mapping = []
            
            for intent in self.intents:
                for pattern in intent["patterns"]:
                    corpus.append(pattern)
                    self.intent_mapping.append(intent["tag"])
            
            # Fit the vectorizer
            self.vectorizer.fit(corpus)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_knowledge_base(self, knowledge_base_path):
        """
        Load knowledge base from a JSON file or use default knowledge base.
        
        Parameters:
        -----------
        knowledge_base_path : str
            Path to a JSON file containing the knowledge base.
            
        Returns:
        --------
        dict
            Knowledge base dictionary.
        """
        default_knowledge_base = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": [
                        "hi", "hello", "hey", "good day", "greetings", "what's up", "how are you"
                    ],
                    "responses": [
                        "Hello! I'm your Scaler career advisor. How can I help you today?",
                        "Hi there! I'm here to help with your tech career questions. What brings you here today?",
                        "Hello! I'm your AI career guide. How can I assist with your tech journey?"
                    ]
                },
                {
                    "tag": "goodbye",
                    "patterns": [
                        "bye", "see you later", "goodbye", "take care", "exit", "end", "quit"
                    ],
                    "responses": [
                        "Goodbye! Feel free to come back if you have more questions.",
                        "Thanks for chatting! Have a great day.",
                        "Bye for now! Don't hesitate to return if you need more guidance."
                    ]
                },
                {
                    "tag": "thanks",
                    "patterns": [
                        "thanks", "thank you", "appreciate it", "helpful", "thanks a lot"
                    ],
                    "responses": [
                        "You're welcome! I'm happy to help.",
                        "Glad I could assist you!",
                        "My pleasure! Let me know if you need anything else."
                    ]
                },
                {
                    "tag": "career_options",
                    "patterns": [
                        "what tech career should I pursue", "best tech fields", "career options", 
                        "which tech field is best", "tech career suggestions",
                        "not sure which field", "career advice", "which path to take"
                    ],
                    "responses": [
                        "There are many exciting tech paths! Could you tell me a bit about your interests and background?",
                        "I'd be happy to help you explore options! What aspects of technology interest you the most?",
                        "Let's find the right tech path for you. What are your strengths and interests?"
                    ]
                },
                {
                    "tag": "course_recommendation",
                    "patterns": [
                        "which course should I take", "recommend a course", "best course for me",
                        "course suggestion", "learning path", "study recommendation",
                        "what should I learn", "course for beginners"
                    ],
                    "responses": [
                        "I can help recommend courses! Could you share what field you're interested in?",
                        "To suggest the right course, I'd need to know your career goals and current skill level.",
                        "I'd be happy to recommend courses! What specific skills are you looking to develop?"
                    ]
                },
                {
                    "tag": "data_science_interest",
                    "patterns": [
                        "interested in data science", "want to learn data science", "become a data scientist",
                        "data analytics", "machine learning", "AI career", "data science path"
                    ],
                    "responses": [
                        "Data Science is an excellent choice! Our Data Science Career Track includes Python, statistics, machine learning, and more.",
                        "For Data Science, I recommend starting with our 'Data Science Fundamentals' course, then advancing to specialized areas.",
                        "With Data Science, you could pursue roles like Data Scientist, ML Engineer, or Data Analyst. Would you like specific course recommendations?"
                    ]
                },
                {
                    "tag": "web_development_interest",
                    "patterns": [
                        "interested in web development", "want to be a web developer", "learn web dev",
                        "frontend", "backend", "full stack", "website creation"
                    ],
                    "responses": [
                        "Web Development is in high demand! Our Web Development tracks cover frontend, backend, or full-stack paths.",
                        "For Web Development, I recommend starting with HTML/CSS/JavaScript, then moving to frameworks like React.",
                        "Would you prefer to focus on frontend (user interfaces), backend (server-side), or full-stack development?"
                    ]
                },
                {
                    "tag": "pricing",
                    "patterns": [
                        "how much does it cost", "course price", "pricing", "payment plans",
                        "course fee", "can I afford", "discount", "financial aid"
                    ],
                    "responses": [
                        "Our courses range from ₹25,000 to ₹2,50,000 depending on the program. We offer EMI options and scholarships too.",
                        "We have flexible payment plans including pay-after-placement options for select courses.",
                        "Would you like me to explain our pricing structure and financing options in detail?"
                    ]
                },
                {
                    "tag": "placement_help",
                    "patterns": [
                        "job placement", "career support", "get a job", "placement guarantee",
                        "interview preparation", "hiring partners", "job assistance"
                    ],
                    "responses": [
                        "Our Career Support includes resume building, interview prep, and connections with 1000+ hiring partners.",
                        "We have a dedicated placement team that works with you until you secure a position.",
                        "Would you like to know more about our placement success rates and partner companies?"
                    ]
                }
            ],
            "courses": [
                {
                    "id": "ds101",
                    "title": "Data Science Fundamentals",
                    "description": "Learn the basics of data analysis, statistics, and Python programming.",
                    "duration": "12 weeks",
                    "level": "beginner",
                    "topics": ["Python", "Statistics", "Data Analysis", "Visualization"],
                    "career_paths": ["Data Analyst", "Junior Data Scientist"]
                },
                {
                    "id": "ds201",
                    "title": "Machine Learning Specialization",
                    "description": "Master machine learning algorithms and their practical applications.",
                    "duration": "16 weeks",
                    "level": "intermediate",
                    "topics": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation"],
                    "career_paths": ["Machine Learning Engineer", "Data Scientist"]
                },
                {
                    "id": "web101",
                    "title": "Web Development Fundamentals",
                    "description": "Build a strong foundation in HTML, CSS, and JavaScript.",
                    "duration": "10 weeks",
                    "level": "beginner",
                    "topics": ["HTML", "CSS", "JavaScript", "Responsive Design"],
                    "career_paths": ["Junior Frontend Developer", "Web Designer"]
                },
                {
                    "id": "web201",
                    "title": "Full-Stack Development with MERN",
                    "description": "Learn to build complete web applications with MongoDB, Express, React, and Node.js.",
                    "duration": "14 weeks",
                    "level": "intermediate",
                    "topics": ["React", "Node.js", "Express", "MongoDB", "RESTful APIs"],
                    "career_paths": ["Full-Stack Developer", "Frontend Developer", "Backend Developer"]
                },
                {
                    "id": "cs101",
                    "title": "Computer Science Fundamentals",
                    "description": "Understand core CS concepts like algorithms, data structures, and OOP.",
                    "duration": "14 weeks",
                    "level": "beginner",
                    "topics": ["Algorithms", "Data Structures", "OOP", "Problem Solving"],
                    "career_paths": ["Software Engineer", "Technical Analyst"]
                }
            ],
            "career_paths": [
                {
                    "name": "Data Scientist",
                    "description": "Analyze and interpret complex data to help organizations make better decisions.",
                    "required_skills": ["Python", "Machine Learning", "Statistics", "SQL", "Data Visualization"],
                    "avg_salary": "₹8,00,000 - ₹18,00,000",
                    "job_outlook": "High growth field with increasing demand across industries."
                },
                {
                    "name": "Full-Stack Developer",
                    "description": "Build complete web applications, working on both client and server software.",
                    "required_skills": ["JavaScript", "React/Angular", "Node.js", "Databases", "HTML/CSS"],
                    "avg_salary": "₹6,00,000 - ₹15,00,000",
                    "job_outlook": "Consistently strong demand in startups and established companies."
                },
                {
                    "name": "Machine Learning Engineer",
                    "description": "Design and implement machine learning models and maintain AI systems.",
                    "required_skills": ["Python", "Deep Learning", "MLOps", "Software Engineering", "Mathematics"],
                    "avg_salary": "₹10,00,000 - ₹20,00,000",
                    "job_outlook": "Rapidly growing field with excellent long-term prospects."
                },
                {
                    "name": "Backend Developer",
                    "description": "Create and maintain server-side applications and databases.",
                    "required_skills": ["Java/Python/Node.js", "Databases", "APIs", "Cloud Services"],
                    "avg_salary": "₹6,00,000 - ₹14,00,000",
                    "job_outlook": "Strong demand across all industries requiring web services."
                },
                {
                    "name": "DevOps Engineer",
                    "description": "Combine software development and operations to improve deployment processes.",
                    "required_skills": ["CI/CD", "Cloud Platforms", "Containerization", "Scripting", "Monitoring"],
                    "avg_salary": "₹8,00,000 - ₹18,00,000",
                    "job_outlook": "Growing demand as more companies adopt DevOps practices."
                }
            ]
        }
        
        if knowledge_base_path:
            try:
                with open(knowledge_base_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading knowledge base: {str(e)}")
                return default_knowledge_base
        else:
            return default_knowledge_base
    
    def preprocess_text(self, text):
        """
        Preprocess text for NLP tasks.
        
        Parameters:
        -----------
        text : str
            Text to preprocess.
            
        Returns:
        --------
        list
            List of lemmatized tokens.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return tokens
    
    def detect_intent(self, text):
        """
        Detect the intent of the user's message.
        
        Parameters:
        -----------
        text : str
            User's message.
            
        Returns:
        --------
        str
            Detected intent tag.
        """
        # If no intents are defined, return None
        if not self.intents:
            return None
        
        # Transform the input text
        text_vector = self.vectorizer.transform([text])
        
        # Create a corpus of all intent patterns
        corpus = []
        intent_mapping = []
        
        for intent in self.intents:
            for pattern in intent["patterns"]:
                corpus.append(pattern)
                intent_mapping.append(intent["tag"])
        
        # Transform the corpus
        corpus_vectors = self.vectorizer.transform(corpus)
        
        # Calculate similarities
        similarities = cosine_similarity(text_vector, corpus_vectors)[0]
        
        # Find the most similar pattern
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        
        # If the best match is below threshold, return None
        if best_match_score < 0.3:
            return None
        
        return intent_mapping[best_match_index]
    
    def get_response(self, intent_tag):
        """
        Get a response based on the detected intent.
        
        Parameters:
        -----------
        intent_tag : str
            Detected intent tag.
            
        Returns:
        --------
        str
            Response to the user.
        """
        # Find the intent with the matching tag
        for intent in self.intents:
            if intent["tag"] == intent_tag:
                # Return a random response from the intent's responses
                return random.choice(intent["responses"])
        
        # If no matching intent is found, return a default response
        return "I'm not sure I understand. Could you please rephrase that?"
    
    def recommend_courses(self, interest=None, level=None):
        """
        Recommend courses based on user interest and level.
        
        Parameters:
        -----------
        interest : str, optional
            User's area of interest.
        level : str, optional
            User's skill level (beginner, intermediate, advanced).
            
        Returns:
        --------
        list
            List of recommended courses.
        """
        recommendations = []
        
        # Filter courses based on interest and level
        for course in self.courses:
            # Check interest
            if interest:
                if interest.lower() not in course["title"].lower() and interest.lower() not in course["description"].lower():
                    interest_match = False
                    for topic in course["topics"]:
                        if interest.lower() in topic.lower():
                            interest_match = True
                            break
                    if not interest_match:
                        continue
            
            # Check level
            if level and course["level"] != level.lower():
                continue
            
            recommendations.append(course)
        
        return recommendations
    
    def suggest_career_path(self, skills=None, interests=None):
        """
        Suggest career paths based on user skills and interests.
        
        Parameters:
        -----------
        skills : list, optional
            List of user skills.
        interests : list, optional
            List of user interests.
            
        Returns:
        --------
        list
            List of suggested career paths.
        """
        suggestions = []
        
        # Convert inputs to lists if they are strings
        if isinstance(skills, str):
            skills = [skills]
        if isinstance(interests, str):
            interests = [interests]
        
        # If no skills or interests are provided, return all career paths
        if not skills and not interests:
            return self.career_paths
        
        # Filter career paths based on skills and interests
        for career in self.career_paths:
            score = 0
            
            # Check skills
            if skills:
                for skill in skills:
                    for required_skill in career["required_skills"]:
                        if skill.lower() in required_skill.lower():
                            score += 1
            
            # Check interests
            if interests:
                for interest in interests:
                    if interest.lower() in career["name"].lower() or interest.lower() in career["description"].lower():
                        score += 2
            
            if score > 0:
                suggestions.append((career, score))
        
        # Sort suggestions by score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the career paths
        return [suggestion[0] for suggestion in suggestions]
    
    def extract_entities(self, text):
        """
        Extract entities like skills, interests, and level from text.
        
        Parameters:
        -----------
        text : str
            User's message.
            
        Returns:
        --------
        dict
            Dictionary of extracted entities.
        """
        entities = {
            "skills": [],
            "interests": [],
            "level": None
        }
        
        # Check for skill keywords
        skill_keywords = ["Python", "JavaScript", "Java", "C++", "HTML", "CSS", "SQL", "machine learning",
                         "data analysis", "web development", "frontend", "backend", "full stack",
                         "cloud", "DevOps", "UI/UX", "mobile", "database", "programming"]
        
        for skill in skill_keywords:
            if skill.lower() in text.lower():
                entities["skills"].append(skill)
        
        # Check for interest areas
        interest_keywords = ["data science", "web development", "machine learning", "artificial intelligence",
                           "frontend", "backend", "full stack", "mobile development", "cloud computing",
                           "DevOps", "cybersecurity", "UI/UX design", "game development"]
        
        for interest in interest_keywords:
            if interest.lower() in text.lower():
                entities["interests"].append(interest)
        
        # Check for level
        if "beginner" in text.lower() or "starting" in text.lower() or "new to" in text.lower():
            entities["level"] = "beginner"
        elif "intermediate" in text.lower() or "some experience" in text.lower():
            entities["level"] = "intermediate"
        elif "advanced" in text.lower() or "experienced" in text.lower() or "expert" in text.lower():
            entities["level"] = "advanced"
        
        return entities
    
    def update_user_profile(self, entities):
        """
        Update the user profile with extracted entities.
        
        Parameters:
        -----------
        entities : dict
            Dictionary of extracted entities.
        """
        if not self.user_profile:
            self.user_profile = {
                "skills": [],
                "interests": [],
                "level": None
            }
        
        # Update skills
        for skill in entities["skills"]:
            if skill not in self.user_profile["skills"]:
                self.user_profile["skills"].append(skill)
        
        # Update interests
        for interest in entities["interests"]:
            if interest not in self.user_profile["interests"]:
                self.user_profile["interests"].append(interest)
        
        # Update level
        if entities["level"]:
            self.user_profile["level"] = entities["level"]
    
    def generate_response(self, user_input):
        """
        Generate a response to the user's input.
        
        Parameters:
        -----------
        user_input : str
            User's message.
            
        Returns:
        --------
        str
            Response to the user.
        """
        # Preprocess the input
        preprocessed_input = ' '.join(self.preprocess_text(user_input))
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "message": user_input})
        
        # Extract entities
        entities = self.extract_entities(user_input)
        
        # Update user profile
        self.update_user_profile(entities)
        
        # Detect intent
        intent = self.detect_intent(preprocessed_input)
        
        # Generate response based on intent
        if intent:
            response = self.get_response(intent)
            
            # Handle specific intents with more dynamic responses
            if intent == "course_recommendation":
                # If we have interests in the user profile, use them to recommend courses
                if self.user_profile.get("interests"):
                    courses = self.recommend_courses(
                        interest=self.user_profile["interests"][0],
                        level=self.user_profile.get("level")
                    )
                    
                    if courses:
                        response = f"Based on your interest in {self.user_profile['interests'][0]}, I recommend these courses:\n\n"
                        for course in courses[:3]:
                            response += f"- {course['title']}: {course['description']} ({course['duration']})\n"
            
            elif intent == "career_options":
                # If we have skills or interests in the user profile, suggest career paths
                if self.user_profile.get("skills") or self.user_profile.get("interests"):
                    careers = self.suggest_career_path(
                        skills=self.user_profile.get("skills"),
                        interests=self.user_profile.get("interests")
                    )
                    
                    if careers:
                        response = "Based on your profile, these career paths might be a good fit:\n\n"
                        for career in careers[:3]:
                            response += f"- {career['name']}: {career['description']}\n  Average salary: {career['avg_salary']}\n"
        else:
            # If no intent is detected, try to provide a helpful response
            if "course" in user_input.lower():
                response = "I offer various courses across data science, web development, and more. Could you specify which area interests you most?"
            elif "career" in user_input.lower() or "job" in user_input.lower():
                response = "There are many exciting tech careers to explore! Could you share your interests or skills so I can provide better guidance?"
            else:
                response = "I'm not sure I understand. Could you please rephrase or provide more details about what you're looking for?"
        
        # Add to conversation history
        self.conversation_history.append({"role": "bot", "message": response})
        
        return response

    def get_conversation_history(self):
        """
        Get the conversation history.
        
        Returns:
        --------
        list
            List of conversation messages.
        """
        return self.conversation_history
    
    def reset_conversation(self):
        """
        Reset the conversation history and user profile.
        """
        self.conversation_history = []
        self.user_profile = {}

# Example usage
if __name__ == "__main__":
    # Create the bot
    bot = CareerAdvisorBot()
    
    print("Career Advisor Bot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Feel free to come back if you have more questions.")
            break
        
        response = bot.generate_response(user_input)
        print(f"Bot: {response}")
