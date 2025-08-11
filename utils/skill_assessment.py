# Interactive Skill Assessment Tool
# This module implements an adaptive skill assessment for educational recommendation

from flask import Flask, request, jsonify
import json
import random
import math
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SkillAssessmentEngine:
    """
    Engine for creating and managing adaptive skill assessments.
    Features progressive difficulty and personalized question selection.
    """
    
    def __init__(self, question_bank_path=None):
        """
        Initialize the skill assessment engine.
        
        Parameters:
        -----------
        question_bank_path : str, optional
            Path to a JSON file containing the question bank.
        """
        # Load question bank
        self.question_bank = self._load_question_bank(question_bank_path)
        
        # Initialize assessment state
        self.current_assessment = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_question_bank(self, question_bank_path):
        """
        Load question bank from a JSON file or use default question bank.
        
        Parameters:
        -----------
        question_bank_path : str
            Path to a JSON file containing the question bank.
            
        Returns:
        --------
        dict
            Question bank dictionary.
        """
        # Default question bank with questions for different domains and difficulty levels
        default_question_bank = {
            "domains": [
                {
                    "name": "data_science",
                    "title": "Data Science",
                    "skills": ["python", "statistics", "machine_learning", "data_visualization", "sql"],
                    "questions": [
                        {
                            "id": "ds_q1",
                            "text": "Which of the following is NOT a Python data structure?",
                            "options": ["List", "Dictionary", "Schema", "Tuple"],
                            "correct_answer": "Schema",
                            "difficulty": 1,
                            "skill": "python"
                        },
                        {
                            "id": "ds_q2",
                            "text": "What does the NumPy function np.array() do?",
                            "options": ["Creates a new array", "Sorts an array", "Filters an array", "Computes array mean"],
                            "correct_answer": "Creates a new array",
                            "difficulty": 1,
                            "skill": "python"
                        },
                        {
                            "id": "ds_q3",
                            "text": "What is the median of the following numbers: 5, 7, 9, 11, 13?",
                            "options": ["5", "7", "9", "11"],
                            "correct_answer": "9",
                            "difficulty": 1,
                            "skill": "statistics"
                        },
                        {
                            "id": "ds_q4",
                            "text": "In statistics, what does p-value represent?",
                            "options": [
                                "The probability that the null hypothesis is true",
                                "The probability of observing data at least as extreme as the test statistic, assuming the null hypothesis is true",
                                "The probability that the alternative hypothesis is true",
                                "The percentage of variance explained by the model"
                            ],
                            "correct_answer": "The probability of observing data at least as extreme as the test statistic, assuming the null hypothesis is true",
                            "difficulty": 3,
                            "skill": "statistics"
                        },
                        {
                            "id": "ds_q5",
                            "text": "Which of the following is NOT a supervised learning algorithm?",
                            "options": ["Linear Regression", "Random Forest", "K-means Clustering", "Support Vector Machine"],
                            "correct_answer": "K-means Clustering",
                            "difficulty": 2,
                            "skill": "machine_learning"
                        },
                        {
                            "id": "ds_q6",
                            "text": "What is the bias-variance tradeoff in machine learning?",
                            "options": [
                                "The balance between model complexity and training error",
                                "The balance between underfitting and overfitting",
                                "The balance between training time and model accuracy",
                                "The balance between feature selection and feature engineering"
                            ],
                            "correct_answer": "The balance between underfitting and overfitting",
                            "difficulty": 3,
                            "skill": "machine_learning"
                        },
                        {
                            "id": "ds_q7",
                            "text": "Which visualization would be most appropriate for showing the distribution of a continuous variable?",
                            "options": ["Bar chart", "Histogram", "Pie chart", "Scatter plot"],
                            "correct_answer": "Histogram",
                            "difficulty": 1,
                            "skill": "data_visualization"
                        },
                        {
                            "id": "ds_q8",
                            "text": "Which SQL keyword is used to filter groups in a GROUP BY clause?",
                            "options": ["WHERE", "HAVING", "FILTER", "GROUP FILTER"],
                            "correct_answer": "HAVING",
                            "difficulty": 2,
                            "skill": "sql"
                        }
                    ]
                },
                {
                    "name": "web_development",
                    "title": "Web Development",
                    "skills": ["html_css", "javascript", "frontend_frameworks", "backend", "databases"],
                    "questions": [
                        {
                            "id": "web_q1",
                            "text": "Which HTML tag is used to create a hyperlink?",
                            "options": ["<link>", "<href>", "<a>", "<url>"],
                            "correct_answer": "<a>",
                            "difficulty": 1,
                            "skill": "html_css"
                        },
                        {
                            "id": "web_q2",
                            "text": "Which CSS property is used to change the text color?",
                            "options": ["text-color", "font-color", "color", "text-style"],
                            "correct_answer": "color",
                            "difficulty": 1,
                            "skill": "html_css"
                        },
                        {
                            "id": "web_q3",
                            "text": "What is the output of: console.log(typeof []);",
                            "options": ["array", "object", "undefined", "null"],
                            "correct_answer": "object",
                            "difficulty": 2,
                            "skill": "javascript"
                        },
                        {
                            "id": "web_q4",
                            "text": "Which JavaScript method is used to add an element to the end of an array?",
                            "options": ["append()", "push()", "add()", "insert()"],
                            "correct_answer": "push()",
                            "difficulty": 1,
                            "skill": "javascript"
                        },
                        {
                            "id": "web_q5",
                            "text": "In React, what is the correct way to pass data from a parent component to a child component?",
                            "options": ["Using context", "Using props", "Using state", "Using redux"],
                            "correct_answer": "Using props",
                            "difficulty": 2,
                            "skill": "frontend_frameworks"
                        },
                        {
                            "id": "web_q6",
                            "text": "What is the purpose of the useEffect hook in React?",
                            "options": [
                                "To handle form submissions",
                                "To create global state",
                                "To perform side effects in function components",
                                "To optimize rendering performance"
                            ],
                            "correct_answer": "To perform side effects in function components",
                            "difficulty": 3,
                            "skill": "frontend_frameworks"
                        },
                        {
                            "id": "web_q7",
                            "text": "Which of the following is NOT a Node.js framework?",
                            "options": ["Express", "Koa", "Django", "Hapi"],
                            "correct_answer": "Django",
                            "difficulty": 1,
                            "skill": "backend"
                        },
                        {
                            "id": "web_q8",
                            "text": "What is a NoSQL database?",
                            "options": [
                                "A database that doesn't use SQL at all",
                                "A database that uses a query language other than SQL",
                                "A database that doesn't follow the relational model",
                                "A database that only stores text data"
                            ],
                            "correct_answer": "A database that doesn't follow the relational model",
                            "difficulty": 2,
                            "skill": "databases"
                        }
                    ]
                }
            ]
        }
        
        if question_bank_path:
            try:
                with open(question_bank_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading question bank: {str(e)}")
                return default_question_bank
        else:
            return default_question_bank
    
    def create_assessment(self, user_id, domain=None, duration=None, difficulty=None):
        """
        Create a new assessment for a user.
        
        Parameters:
        -----------
        user_id : str
            Unique identifier for the user.
        domain : str, optional
            Domain for the assessment (e.g., 'data_science', 'web_development').
        duration : int, optional
            Duration of the assessment in minutes.
        difficulty : int, optional
            Initial difficulty level (1-5).
            
        Returns:
        --------
        dict
            Assessment information.
        """
        # Find the domain in question bank
        domain_data = None
        if domain:
            for d in self.question_bank["domains"]:
                if d["name"] == domain:
                    domain_data = d
                    break
        
        # If domain not found or not specified, use the first domain
        if not domain_data:
            domain_data = self.question_bank["domains"][0]
        
        # Set initial difficulty
        difficulty = difficulty if difficulty else 1
        
        # Create the assessment
        assessment = {
            "user_id": user_id,
            "domain": domain_data["name"],
            "domain_title": domain_data["title"],
            "start_time": pd.Timestamp.now().isoformat(),
            "duration": duration if duration else 20,  # Default: 20 minutes
            "questions": [],
            "current_question_index": 0,
            "current_difficulty": difficulty,
            "skill_scores": {skill: 0 for skill in domain_data["skills"]},
            "completed": False
        }
        
        # Select initial questions
        assessment["questions"] = self._select_initial_questions(domain_data, difficulty)
        
        # Store the assessment
        self.current_assessment = assessment
        
        return {
            "assessment_id": user_id,
            "domain": assessment["domain_title"],
            "duration": assessment["duration"],
            "total_questions": len(assessment["questions"]),
            "first_question": self._get_question_data(assessment["questions"][0])
        }
    
    def _select_initial_questions(self, domain_data, difficulty, num_questions=10):
        """
        Select initial questions for the assessment.
        
        Parameters:
        -----------
        domain_data : dict
            Domain data from question bank.
        difficulty : int
            Initial difficulty level.
        num_questions : int
            Number of questions to select.
            
        Returns:
        --------
        list
            Selected questions.
        """
        all_questions = domain_data["questions"]
        skills = domain_data["skills"]
        
        # Filter questions by difficulty (allow +/- 1 level)
        valid_difficulties = [max(1, difficulty - 1), difficulty, min(5, difficulty + 1)]
        filtered_questions = [q for q in all_questions if q.get("difficulty", 1) in valid_difficulties]
        
        # If not enough questions, use all questions
        if len(filtered_questions) < num_questions:
            filtered_questions = all_questions
        
        # Try to select questions from each skill
        selected_questions = []
        questions_per_skill = math.ceil(num_questions / len(skills))
        
        for skill in skills:
            skill_questions = [q for q in filtered_questions if q.get("skill") == skill]
            selected = random.sample(skill_questions, min(questions_per_skill, len(skill_questions)))
            selected_questions.extend(selected)
        
        # If still need more questions, randomly select from remaining
        remaining_count = num_questions - len(selected_questions)
        if remaining_count > 0:
            remaining_questions = [q for q in filtered_questions if q not in selected_questions]
            if remaining_questions:
                additional = random.sample(remaining_questions, min(remaining_count, len(remaining_questions)))
                selected_questions.extend(additional)
        
        # Trim if we have too many
        if len(selected_questions) > num_questions:
            selected_questions = random.sample(selected_questions, num_questions)
        
        # Shuffle the questions
        random.shuffle(selected_questions)
        
        return selected_questions
    
    def _get_question_data(self, question):
        """
        Extract question data for presentation.
        
        Parameters:
        -----------
        question : dict
            Question data.
            
        Returns:
        --------
        dict
            Question data for presentation.
        """
        return {
            "id": question["id"],
            "text": question["text"],
            "options": question["options"],
            "skill": question.get("skill")
        }
    
    def submit_answer(self, question_id, answer):
        """
        Submit an answer for a question.
        
        Parameters:
        -----------
        question_id : str
            ID of the question.
        answer : str
            User's answer.
            
        Returns:
        --------
        dict
            Result of the submission and next question if available.
        """
        if not self.current_assessment:
            return {"error": "No active assessment"}
        
        # Find the current question
        current_question = self.current_assessment["questions"][self.current_assessment["current_question_index"]]
        
        # Verify question ID
        if current_question["id"] != question_id:
            return {"error": f"Expected answer for question {current_question['id']}, got {question_id}"}
        
        # Check the answer
        is_correct = answer == current_question["correct_answer"]
        
        # Update skill score
        skill = current_question.get("skill")
        if skill:
            # Award points based on difficulty and correctness
            difficulty = current_question.get("difficulty", 1)
            points = difficulty if is_correct else -1
            self.current_assessment["skill_scores"][skill] += points
        
        # Update difficulty based on answer
        if is_correct:
            self.current_assessment["current_difficulty"] = min(5, self.current_assessment["current_difficulty"] + 0.5)
        else:
            self.current_assessment["current_difficulty"] = max(1, self.current_assessment["current_difficulty"] - 0.5)
        
        # Move to next question
        self.current_assessment["current_question_index"] += 1
        
        # Check if assessment is complete
        if self.current_assessment["current_question_index"] >= len(self.current_assessment["questions"]):
            self.current_assessment["completed"] = True
            return {
                "is_correct": is_correct,
                "correct_answer": current_question["correct_answer"],
                "assessment_complete": True,
                "results": self._generate_assessment_results()
            }
        
        # Return result and next question
        next_question = self.current_assessment["questions"][self.current_assessment["current_question_index"]]
        
        return {
            "is_correct": is_correct,
            "correct_answer": current_question["correct_answer"],
            "next_question": self._get_question_data(next_question),
            "progress": {
                "current": self.current_assessment["current_question_index"],
                "total": len(self.current_assessment["questions"])
            }
        }
    
    def _generate_assessment_results(self):
        """
        Generate results for a completed assessment.
        
        Returns:
        --------
        dict
            Assessment results.
        """
        if not self.current_assessment or not self.current_assessment["completed"]:
            return {"error": "Assessment not complete"}
        
        # Calculate overall score
        total_questions = len(self.current_assessment["questions"])
        correct_answers = 0
        
        for i, question in enumerate(self.current_assessment["questions"]):
            if i < self.current_assessment["current_question_index"]:
                # This assumes the answer was already checked and skill score updated
                skill = question.get("skill")
                if skill and self.current_assessment["skill_scores"][skill] > 0:
                    correct_answers += 1
        
        overall_score = (correct_answers / total_questions) * 100
        
        # Normalize skill scores
        skill_scores = {}
        for skill, score in self.current_assessment["skill_scores"].items():
            # Count questions for this skill
            skill_questions = [q for q in self.current_assessment["questions"] if q.get("skill") == skill]
            max_possible = sum([q.get("difficulty", 1) for q in skill_questions])
            
            if max_possible > 0:
                # Normalize to 0-100
                normalized_score = max(0, min(100, (score / max_possible) * 100))
                skill_scores[skill] = normalized_score
            else:
                skill_scores[skill] = 0
        
        # Generate course recommendations based on skill scores
        recommendations = self._generate_recommendations(skill_scores)
        
        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "strengths": self._identify_strengths(skill_scores),
            "weaknesses": self._identify_weaknesses(skill_scores),
            "recommendations": recommendations
        }
    
    def _identify_strengths(self, skill_scores, threshold=70):
        """
        Identify strengths based on skill scores.
        
        Parameters:
        -----------
        skill_scores : dict
            Dictionary of skill scores.
        threshold : float
            Threshold for considering a skill as strength.
            
        Returns:
        --------
        list
            List of strengths.
        """
        return [skill for skill, score in skill_scores.items() if score >= threshold]
    
    def _identify_weaknesses(self, skill_scores, threshold=40):
        """
        Identify weaknesses based on skill scores.
        
        Parameters:
        -----------
        skill_scores : dict
            Dictionary of skill scores.
        threshold : float
            Threshold for considering a skill as weakness.
            
        Returns:
        --------
        list
            List of weaknesses.
        """
        return [skill for skill, score in skill_scores.items() if score < threshold]
    
    def _generate_recommendations(self, skill_scores):
        """
        Generate course recommendations based on skill scores.
        
        Parameters:
        -----------
        skill_scores : dict
            Dictionary of skill scores.
            
        Returns:
        --------
        list
            List of course recommendations.
        """
        # This would normally be a more sophisticated recommendation system
        # For now, we'll use a simple rule-based approach
        
        # Sample course data (would typically come from a database)
        courses = [
            {
                "id": "python101",
                "title": "Python Fundamentals",
                "description": "Master the basics of Python programming",
                "target_skills": ["python"],
                "level": "beginner"
            },
            {
                "id": "ds101",
                "title": "Data Science Fundamentals",
                "description": "Learn the basics of data analysis and statistics",
                "target_skills": ["statistics", "data_visualization"],
                "level": "beginner"
            },
            {
                "id": "ml101",
                "title": "Introduction to Machine Learning",
                "description": "Learn the fundamentals of machine learning algorithms",
                "target_skills": ["machine_learning"],
                "level": "intermediate"
            },
            {
                "id": "sql101",
                "title": "SQL for Data Analysis",
                "description": "Master database queries for data analysis",
                "target_skills": ["sql"],
                "level": "beginner"
            },
            {
                "id": "web101",
                "title": "Web Development Fundamentals",
                "description": "Learn the basics of HTML, CSS, and JavaScript",
                "target_skills": ["html_css", "javascript"],
                "level": "beginner"
            },
            {
                "id": "react101",
                "title": "React Fundamentals",
                "description": "Build interactive UIs with React",
                "target_skills": ["javascript", "frontend_frameworks"],
                "level": "intermediate"
            },
            {
                "id": "node101",
                "title": "Node.js Fundamentals",
                "description": "Learn server-side JavaScript with Node.js",
                "target_skills": ["javascript", "backend"],
                "level": "intermediate"
            },
            {
                "id": "db101",
                "title": "Database Design",
                "description": "Learn relational and NoSQL database design",
                "target_skills": ["databases"],
                "level": "intermediate"
            }
        ]
        
        recommendations = []
        weaknesses = self._identify_weaknesses(skill_scores)
        
        # First, recommend courses for weak skills
        for skill in weaknesses:
            relevant_courses = [c for c in courses if skill in c["target_skills"]]
            if relevant_courses:
                # Sort by level: beginner courses first for weak skills
                beginner_courses = [c for c in relevant_courses if c["level"] == "beginner"]
                if beginner_courses:
                    recommendations.append(beginner_courses[0])
        
        # Then, recommend advanced courses for strong skills
        strengths = self._identify_strengths(skill_scores)
        for skill in strengths:
            relevant_courses = [
                c for c in courses 
                if skill in c["target_skills"] and c["level"] in ["intermediate", "advanced"]
            ]
            if relevant_courses:
                for course in relevant_courses:
                    if course not in recommendations:
                        recommendations.append(course)
                        break
        
        # Limit to top 3 recommendations
        return recommendations[:3]
    
    def get_assessment_progress(self):
        """
        Get the current assessment progress.
        
        Returns:
        --------
        dict
            Assessment progress information.
        """
        if not self.current_assessment:
            return {"error": "No active assessment"}
        
        return {
            "domain": self.current_assessment["domain_title"],
            "current_question": self.current_assessment["current_question_index"] + 1,
            "total_questions": len(self.current_assessment["questions"]),
            "completed": self.current_assessment["completed"]
        }
    
    def end_assessment(self):
        """
        End the current assessment prematurely.
        
        Returns:
        --------
        dict
            Assessment results.
        """
        if not self.current_assessment:
            return {"error": "No active assessment"}
        
        self.current_assessment["completed"] = True
        return self._generate_assessment_results()

# Flask API for the Skill Assessment Engine
app = Flask(__name__)
assessment_engine = SkillAssessmentEngine()

@app.route('/assessment/create', methods=['POST'])
def create_assessment():
    data = request.json
    user_id = data.get('user_id')
    domain = data.get('domain')
    duration = data.get('duration')
    difficulty = data.get('difficulty')
    
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    result = assessment_engine.create_assessment(user_id, domain, duration, difficulty)
    return jsonify(result)

@app.route('/assessment/submit', methods=['POST'])
def submit_answer():
    data = request.json
    question_id = data.get('question_id')
    answer = data.get('answer')
    
    if not question_id or not answer:
        return jsonify({"error": "question_id and answer are required"}), 400
    
    result = assessment_engine.submit_answer(question_id, answer)
    return jsonify(result)

@app.route('/assessment/progress', methods=['GET'])
def get_progress():
    result = assessment_engine.get_assessment_progress()
    return jsonify(result)

@app.route('/assessment/end', methods=['POST'])
def end_assessment():
    result = assessment_engine.end_assessment()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
