# Email Content Generator
# This module uses OpenAI GPT to generate personalized emails for cold leads

import os
import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Optional
import logging
import time

class EmailContentGenerator:
    """
    Uses OpenAI's API to generate personalized email content for cold leads.
    Implements caching and rate limiting to optimize API usage.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the EmailContentGenerator.
        
        Parameters:
        -----------
        api_key : str
            OpenAI API key
        model : str
            OpenAI model to use (default: gpt-3.5-turbo)
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_email(self, lead_data: Dict, email_type: str) -> Dict:
        """
        Generate a personalized email based on lead data.
        
        Parameters:
        -----------
        lead_data : Dict
            Dictionary containing lead information.
        email_type : str
            Type of email to generate (initial_contact, follow_up, final_offer).
            
        Returns:
        --------
        Dict
            Dictionary containing subject and body of the email.
        """
        # Create cache key
        cache_key = f"{lead_data['id']}_{email_type}"
        
        # Check if this request is cached
        if cache_key in self.cache:
            self.logger.info(f"Using cached email for lead {lead_data['id']}")
            return self.cache[cache_key]
        
        # Construct prompt based on email type
        prompt = self._construct_prompt(lead_data, email_type)
        
        # Call OpenAI API with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._call_openai_api(prompt)
                break
            except Exception as e:
                self.logger.error(f"Error calling OpenAI API: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse response
        email_content = self._parse_response(response)
        
        # Cache the result
        self.cache[cache_key] = email_content
        
        return email_content
    
    def generate_batch_emails(self, leads_data: List[Dict], email_type: str) -> List[Dict]:
        """
        Generate personalized emails for multiple leads.
        
        Parameters:
        -----------
        leads_data : List[Dict]
            List of dictionaries containing lead information.
        email_type : str
            Type of email to generate.
            
        Returns:
        --------
        List[Dict]
            List of dictionaries containing subject and body for each lead.
        """
        results = []
        
        for lead_data in leads_data:
            try:
                email = self.generate_email(lead_data, email_type)
                results.append({
                    'lead_id': lead_data['id'],
                    'email': email
                })
            except Exception as e:
                self.logger.error(f"Error generating email for lead {lead_data['id']}: {str(e)}")
                results.append({
                    'lead_id': lead_data['id'],
                    'error': str(e)
                })
        
        return results
    
    def _construct_prompt(self, lead_data: Dict, email_type: str) -> str:
        """
        Construct a prompt for the OpenAI API based on lead data and email type.
        
        Parameters:
        -----------
        lead_data : Dict
            Dictionary containing lead information.
        email_type : str
            Type of email to generate.
            
        Returns:
        --------
        str
            Prompt for the OpenAI API.
        """
        # Extract basic lead information
        name = lead_data.get('name', 'there')
        interest_area = lead_data.get('interest_area', 'technology courses')
        last_interaction = lead_data.get('last_interaction_date', 'some time ago')
        location = lead_data.get('location', 'your area')
        
        # Format time since last interaction in a human-readable way
        if isinstance(last_interaction, str):
            time_since = "some time ago"
        else:
            # Calculate days since last interaction
            time_since = f"{(pd.Timestamp.now() - pd.Timestamp(last_interaction)).days} days ago"
        
        # Construct system message based on email type
        if email_type == 'initial_contact':
            system_message = (
                "You are an expert email content writer for an educational technology company called Scaler. "
                "Your task is to write a personalized re-engagement email to a cold lead who previously "
                "showed interest in our courses but never completed enrollment. The email should be "
                "warm, professional, and focused on providing value rather than being sales-y. "
                "The email should include a clear subject line and body text."
            )
            
            # Construct user message with lead details
            user_message = f"""
            Please write a personalized re-engagement email for a lead with the following details:
            
            Name: {name}
            Primary interest: {interest_area}
            Last interaction: {time_since}
            Location: {location}
            
            Additional context:
            - This is our first re-engagement attempt after a period of inactivity
            - The email should offer something valuable (report, insight, free resource) related to their interest
            - Include a soft call-to-action for a next step (but not pushing for immediate enrollment)
            - The email should be conversational and personal in tone
            - Keep the email concise (150-200 words maximum)
            
            Format the response as:
            SUBJECT: [The subject line]
            
            [The email body]
            """
        
        elif email_type == 'follow_up':
            system_message = (
                "You are an expert email content writer for an educational technology company called Scaler. "
                "Your task is to write a follow-up email to a cold lead who received our initial re-engagement "
                "email but didn't take action. The email should build on the previous communication, "
                "provide additional value, and encourage engagement."
            )
            
            # Get additional data for follow-up
            opened_first_email = lead_data.get('opened_first_email', False)
            clicked_links = lead_data.get('clicked_links', False)
            
            # Construct user message with lead details
            user_message = f"""
            Please write a follow-up email for a lead with the following details:
            
            Name: {name}
            Primary interest: {interest_area}
            Last interaction: {time_since}
            Location: {location}
            Opened previous email: {'Yes' if opened_first_email else 'No'}
            Clicked links in previous email: {'Yes' if clicked_links else 'No'}
            
            Additional context:
            - This is a follow-up to our initial re-engagement email
            - {'Since they opened the previous email but didn't click, focus on making the value proposition clearer' if opened_first_email and not clicked_links else ''}
            - {'Since they didn't open the previous email, use a completely different subject line and approach' if not opened_first_email else ''}
            - {'Since they clicked but didn't take further action, focus on addressing potential objections or barriers' if opened_first_email and clicked_links else ''}
            - Include a more direct call-to-action, such as booking a consultation call
            - Keep the email concise (150-200 words maximum)
            
            Format the response as:
            SUBJECT: [The subject line]
            
            [The email body]
            """
        
        elif email_type == 'final_offer':
            system_message = (
                "You are an expert email content writer for an educational technology company called Scaler. "
                "Your task is to write a final offer email to a cold lead who has shown some engagement with "
                "our previous communications but hasn't converted. This email should create urgency "
                "and present a special, limited-time offer to encourage immediate action."
            )
            
            # Get additional data for final offer
            engagement_level = lead_data.get('engagement_level', 'low')
            viewed_courses = lead_data.get('viewed_courses', [])
            price_sensitivity = lead_data.get('price_sensitivity', 'medium')
            
            # Construct user message with lead details
            user_message = f"""
            Please write a final offer email for a lead with the following details:
            
            Name: {name}
            Primary interest: {interest_area}
            Last interaction: {time_since}
            Location: {location}
            Engagement level: {engagement_level}
            Viewed courses: {', '.join(viewed_courses) if viewed_courses else 'None'}
            Price sensitivity: {price_sensitivity}
            
            Additional context:
            - This is our final offer email in the re-engagement sequence
            - {'Since they appear price-sensitive, emphasize the ROI and payment options' if price_sensitivity in ['high', 'medium'] else ''}
            - {'Mention the specific courses they viewed and any special offer related to those' if viewed_courses else ''}
            - Include a time-limited special offer (e.g., discount, additional bonuses, or extended payment plan)
            - Create a sense of urgency but remain professional (avoid excessive pressure tactics)
            - Include a very clear call-to-action
            - Keep the email concise but comprehensive (200-250 words)
            
            Format the response as:
            SUBJECT: [The subject line]
            
            [The email body]
            """
        
        else:
            raise ValueError(f"Unknown email type: {email_type}")
        
        return {
            "system_message": system_message,
            "user_message": user_message
        }
    
    def _call_openai_api(self, prompt: Dict) -> str:
        """
        Call the OpenAI API with the constructed prompt.
        
        Parameters:
        -----------
        prompt : Dict
            Dictionary containing system_message and user_message.
            
        Returns:
        --------
        str
            Response from the OpenAI API.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system_message"]},
                    {"role": "user", "content": prompt["user_message"]}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API Error: {str(e)}")
            raise
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the OpenAI API response into subject and body.
        
        Parameters:
        -----------
        response : str
            Response from the OpenAI API.
            
        Returns:
        --------
        Dict
            Dictionary containing subject and body of the email.
        """
        try:
            # Extract subject line
            subject_start = response.find("SUBJECT:") + 8
            subject_end = response.find("\n", subject_start)
            subject = response[subject_start:subject_end].strip()
            
            # Extract email body
            body = response[subject_end:].strip()
            
            return {
                "subject": subject,
                "body": body
            }
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            # Fallback parsing
            parts = response.split("\n", 1)
            if len(parts) > 1 and "SUBJECT:" in parts[0]:
                subject = parts[0].replace("SUBJECT:", "").strip()
                body = parts[1].strip()
            else:
                # If we can't parse it properly, make a best effort
                subject = "Important information about your Scaler journey"
                body = response
            
            return {
                "subject": subject,
                "body": body
            }

# Example usage
if __name__ == "__main__":
    # This would be loaded from environment variables in production
    api_key = "your-openai-api-key"
    
    # Create the generator
    generator = EmailContentGenerator(api_key)
    
    # Sample lead data
    lead_data = {
        'id': '12345',
        'name': 'Alex Johnson',
        'interest_area': 'Data Science',
        'last_interaction_date': '2024-11-15',
        'location': 'Bangalore',
        'opened_first_email': True,
        'clicked_links': False,
        'engagement_level': 'medium',
        'viewed_courses': ['Python Foundations', 'Machine Learning Basics'],
        'price_sensitivity': 'high'
    }
    
    # Generate an initial contact email
    initial_email = generator.generate_email(lead_data, 'initial_contact')
    print("INITIAL CONTACT EMAIL:")
    print(f"Subject: {initial_email['subject']}")
    print(f"Body: {initial_email['body']}")
    
    # Generate a follow-up email
    follow_up_email = generator.generate_email(lead_data, 'follow_up')
    print("\nFOLLOW-UP EMAIL:")
    print(f"Subject: {follow_up_email['subject']}")
    print(f"Body: {follow_up_email['body']}")
    
    # Generate a final offer email
    final_offer_email = generator.generate_email(lead_data, 'final_offer')
    print("\nFINAL OFFER EMAIL:")
    print(f"Subject: {final_offer_email['subject']}")
    print(f"Body: {final_offer_email['body']}")
