import asyncio
import sys
import os
import time
import random
from datetime import datetime, timezone
from typing import Optional


# Ensure src directory is in Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from core.config_loader import ConfigLoader
from core.browser_manager import BrowserManager
from core.llm_service import LLMService
from utils.file_handler import FileHandler
from data_models import AccountConfig, TweetContent, LLMSettings, ScrapedTweet, ActionConfig
from features.scraper import TweetScraper
from features.publisher import TweetPublisher
from features.engagement import TweetEngagement
from features.analyzer import TweetAnalyzer
import logging

PROMPT_TEMPLATES = {
    "suggest_alternatives": """
    You are a helpful assistant that suggests alternatives to companies mentioned in X posts.
    Given the following post: "{tweet_text}"
    The company mentioned is {company}.
    Suggest some alternatives to {company} in a friendly and helpful manner, considering the context of the post.
    Keep the response concise and suitable for an X reply (max 280 characters).
    """
}

# Initialize main config loader and logger
main_config_loader = ConfigLoader()
setup_logger(main_config_loader)
logger = logging.getLogger(__name__)

class TwitterOrchestrator:
    def __init__(self):
        self.config_loader = main_config_loader
        self.file_handler = FileHandler(self.config_loader)
        self.global_settings = self.config_loader.get_settings()
        self.accounts_data = self.config_loader.get_accounts_config()
        
        self.processed_action_keys = self.file_handler.load_processed_action_keys() # Load processed action keys
    
    def generate_alternative_reply(self, tweet: ScrapedTweet, account: AccountConfig) -> Optional[str]:
        """Generate a reply suggesting alternatives to the target company."""
        company = next((c for c in account.target_companies if f"@{c.lower()}" in tweet.text_content.lower()), None)
        if not company:
            logger.debug(f"[{account.account_id}] No target company found in tweet {tweet.tweet_id}")
            return None
        prompt_template = PROMPT_TEMPLATES.get(account.reply_settings.get("prompt_template", ""), "")
        if not prompt_template:
            logger.error(f"[{account.account_id}] No prompt template found for {account.reply_settings.get('prompt_template')}")
            return None
        prompt = prompt_template.format(tweet_text=tweet.text_content, company=company)
        llm_service = LLMService(config_loader=self.config_loader)
        reply = llm_service.generate_text(
            prompt,
            service_preference=account.reply_settings.get("llm_service", "openai"),
            model_name=account.llm_settings_override.model_name_override if account.llm_settings_override else None,
            max_tokens=150,
            temperature=0.7
        )
        return reply

    async def _process_account(self, account_dict: dict):
        """Processes tasks for a single Twitter account."""
        try:
            account = AccountConfig.model_validate(account_dict)
        except Exception as e:
            logger.error(f"Failed to parse account configuration for {account_dict.get('account_id', 'UnknownAccount')}: {e}. Skipping account.")
            return

        if not account.is_active:
            logger.info(f"Account {account.account_id} is inactive. Skipping.")
            return

        logger.info(f"--- Starting processing for account: {account.account_id} ---")
        
        browser_manager = None
        try:
            browser_manager = BrowserManager(account_config=account_dict)
            llm_service = LLMService(config_loader=self.config_loader)
            
            scraper = TweetScraper(browser_manager, account_id=account.account_id)
            publisher = TweetPublisher(browser_manager, llm_service, account)
            engagement = TweetEngagement(browser_manager, account)
            
            automation_settings = self.global_settings.get('twitter_automation', {})
            global_action_config_dict = automation_settings.get('action_config', {})
            current_action_config = account.action_config or ActionConfig(**global_action_config_dict)
            
            analyzer = TweetAnalyzer(llm_service, account_config=account)
            
            llm_for_post = account.llm_settings_override or current_action_config.llm_settings_for_post
            llm_for_reply = account.llm_settings_override or current_action_config.llm_settings_for_reply
            llm_for_thread_analysis = account.llm_settings_override or current_action_config.llm_settings_for_thread_analysis
            
            # Existing Action 1: Competitor reposts
            competitor_profiles_for_account = account.competitor_profiles
            if current_action_config.enable_competitor_reposts and competitor_profiles_for_account:
                logger.info(f"[{account.account_id}] Starting competitor profile scraping and posting using {len(competitor_profiles_for_account)} profiles.")
                for profile_url in competitor_profiles_for_account:
                    logger.info(f"[{account.account_id}] Scraping profile: {str(profile_url)}")
                    tweets_from_profile = await asyncio.to_thread(
                        scraper.scrape_tweets_from_profile,
                        str(profile_url),
                        max_tweets=current_action_config.max_posts_per_competitor_run * 3
                    )
                    posts_made_this_profile = 0
                    for scraped_tweet in tweets_from_profile:
                        if posts_made_this_profile >= current_action_config.max_posts_per_competitor_run:
                            break
                        if current_action_config.repost_only_tweets_with_media and not scraped_tweet.embedded_media_urls:
                            continue
                        if scraped_tweet.like_count < current_action_config.min_likes_for_repost_candidate:
                            continue
                        if scraped_tweet.retweet_count < current_action_config.min_retweets_for_repost_candidate:
                            continue
                        interaction_type = current_action_config.competitor_post_interaction_type
                        action_key = f"{interaction_type}_{account.account_id}_{scraped_tweet.tweet_id}"
                        if action_key in self.processed_action_keys:
                            continue
                        if scraped_tweet.is_thread_candidate and current_action_config.enable_thread_analysis:
                            is_confirmed = await analyzer.check_if_thread_with_llm(scraped_tweet, custom_llm_settings=llm_for_thread_analysis)
                            scraped_tweet.is_confirmed_thread = is_confirmed
                        interaction_success = False
                        if interaction_type == "repost":
                            prompt = f"Rewrite this tweet in an engaging way: '{scraped_tweet.text_content}' by {scraped_tweet.user_handle or 'a user'}."
                            if scraped_tweet.is_confirmed_thread:
                                prompt = f"This tweet is part of a thread. Rewrite its essence engagingly: '{scraped_tweet.text_content}' by {scraped_tweet.user_handle or 'a user'}."
                            new_tweet_content = TweetContent(text=prompt)
                            interaction_success = await publisher.post_new_tweet(new_tweet_content, llm_settings=llm_for_post)
                        elif interaction_type == "retweet":
                            interaction_success = await publisher.retweet_tweet(scraped_tweet)
                        elif interaction_type == "quote_tweet":
                            quote_prompt = current_action_config.prompt_for_quote_tweet_from_competitor.format(
                                user_handle=(scraped_tweet.user_handle or "a user"),
                                tweet_text=scraped_tweet.text_content
                            )
                            interaction_success = await publisher.retweet_tweet(scraped_tweet, quote_text_prompt_or_direct=quote_prompt, llm_settings_for_quote=llm_for_post)
                        if interaction_success:
                            self.file_handler.save_processed_action_key(action_key, timestamp=datetime.now().isoformat())
                            self.processed_action_keys.add(action_key)
                            posts_made_this_profile += 1
                            await asyncio.sleep(random.uniform(current_action_config.min_delay_between_actions_seconds, current_action_config.max_delay_between_actions_seconds))
            
            # Existing Action 2: Keyword replies
            target_keywords_for_account = account.target_keywords
            if current_action_config.enable_keyword_replies and target_keywords_for_account:
                for keyword in target_keywords_for_account:
                    tweets_for_keyword = await asyncio.to_thread(
                        scraper.scrape_tweets_by_keyword,
                        keyword,
                        max_tweets=current_action_config.max_replies_per_keyword_run * 2
                    )
                    replies_made_this_keyword = 0
                    for scraped_tweet_to_reply in tweets_for_keyword:
                        if replies_made_this_keyword >= current_action_config.max_replies_per_keyword_run:
                            break
                        action_key = f"reply_{account.account_id}_{scraped_tweet_to_reply.tweet_id}"
                        if action_key in self.processed_action_keys:
                            continue
                        if current_action_config.avoid_replying_to_own_tweets and scraped_tweet_to_reply.user_handle and account.account_id.lower() in scraped_tweet_to_reply.user_handle.lower():
                            continue
                        if current_action_config.reply_only_to_recent_tweets_hours and scraped_tweet_to_reply.created_at:
                            now_utc = datetime.now(timezone.utc)
                            tweet_age_hours = (now_utc - scraped_tweet_to_reply.created_at).total_seconds() / 3600
                            if tweet_age_hours > current_action_config.reply_only_to_recent_tweets_hours:
                                continue
                        if scraped_tweet_to_reply.is_thread_candidate and current_action_config.enable_thread_analysis:
                            is_confirmed = await analyzer.check_if_thread_with_llm(scraped_tweet_to_reply, custom_llm_settings=llm_for_thread_analysis)
                            scraped_tweet_to_reply.is_confirmed_thread = is_confirmed
                        reply_prompt_context = "This tweet is part of a thread." if scraped_tweet_to_reply.is_confirmed_thread else "This is a standalone tweet."
                        reply_prompt = f"Generate an insightful and engaging reply to the following tweet. {reply_prompt_context}\n\nOriginal tweet by @{scraped_tweet_to_reply.user_handle or 'user'}:\n\"{scraped_tweet_to_reply.text_content}\"\n\nYour reply:"
                        generated_reply_text = await llm_service.generate_text(
                            prompt=reply_prompt,
                            service_preference=llm_for_reply.service_preference,
                            model_name=llm_for_reply.model_name_override,
                            max_tokens=llm_for_reply.max_tokens,
                            temperature=llm_for_reply.temperature
                        )
                        if generated_reply_text and await publisher.reply_to_tweet(scraped_tweet_to_reply, generated_reply_text):
                            self.file_handler.save_processed_action_key(action_key, timestamp=datetime.now().isoformat())
                            self.processed_action_keys.add(action_key)
                            replies_made_this_keyword += 1
                            await asyncio.sleep(random.uniform(current_action_config.min_delay_between_actions_seconds, current_action_config.max_delay_between_actions_seconds))
            
            # New Action: Scrape mentions and reply with alternatives
            if account.reply_settings and account.reply_settings.get("trigger") == "mentions":
                mentions = account.reply_settings.get("mentions", [])
                for mention in mentions:
                    logger.info(f"[{account.account_id}] Scraping tweets mentioning @{mention}")
                    tweets = await asyncio.to_thread(
                        scraper.scrape_tweets_by_mention,
                        mention,
                        max_tweets=10
                    )
                    for tweet in tweets:
                        reply_text = self.generate_alternative_reply(tweet, account)
                        if reply_text:
                            action_key = f"reply_{account.account_id}_{tweet.tweet_id}"
                            if action_key in self.processed_action_keys:
                                logger.info(f"[{account.account_id}] Already replied to tweet {tweet.tweet_id}. Skipping.")
                                continue
                            success = await publisher.reply_to_tweet(tweet, reply_text)
                            if success:
                                self.file_handler.save_processed_action_key(action_key, timestamp=datetime.now().isoformat())
                                self.processed_action_keys.add(action_key)
                                logger.info(f"[{account.account_id}] Successfully replied to tweet {tweet.tweet_id}")
                            else:
                                logger.error(f"[{account.account_id}] Failed to reply to tweet {tweet.tweet_id}")
                            await asyncio.sleep(random.uniform(60, 180))
            

            # Action 3: Scrape news/research sites and post summaries/links
            news_sites_for_account = account.news_sites
            research_sites_for_account = account.research_paper_sites
            if current_action_config.enable_content_curation_posts and (news_sites_for_account or research_sites_for_account):
                 logger.info(f"[{account.account_id}] Content curation from news/research sites is planned.")
            elif current_action_config.enable_content_curation_posts:
                logger.info(f"[{account.account_id}] Content curation enabled, but no news/research sites configured for this account.")


            # Action 4: Like tweets
            if current_action_config.enable_liking_tweets:
                keywords_to_like = current_action_config.like_tweets_from_keywords or []
                if keywords_to_like:
                    logger.info(f"[{account.account_id}] Starting to like tweets based on {len(keywords_to_like)} keywords.")
                    likes_done_this_run = 0
                    for keyword in keywords_to_like:
                        if likes_done_this_run >= current_action_config.max_likes_per_run:
                            break
                        logger.info(f"[{account.account_id}] Searching for tweets with keyword '{keyword}' to like.")
                        tweets_to_potentially_like = await asyncio.to_thread(
                            scraper.scrape_tweets_by_keyword,
                            keyword,
                            max_tweets=current_action_config.max_likes_per_run * 2 # Fetch more to have options
                        )
                        for tweet_to_like in tweets_to_potentially_like:
                            if likes_done_this_run >= current_action_config.max_likes_per_run:
                                break
                            
                            action_key = f"like_{account.account_id}_{tweet_to_like.tweet_id}"
                            if action_key in self.processed_action_keys:
                                logger.info(f"[{account.account_id}] Already liked or processed tweet {tweet_to_like.tweet_id}. Skipping.")
                                continue
                            
                            if current_action_config.avoid_replying_to_own_tweets and tweet_to_like.user_handle and account.account_id.lower() in tweet_to_like.user_handle.lower():
                                logger.info(f"[{account.account_id}] Skipping own tweet {tweet_to_like.tweet_id} for liking.")
                                continue

                            logger.info(f"[{account.account_id}] Attempting to like tweet {tweet_to_like.tweet_id} from URL: {tweet_to_like.tweet_url}")
                            like_success = await engagement.like_tweet(tweet_id=tweet_to_like.tweet_id, tweet_url=str(tweet_to_like.tweet_url) if tweet_to_like.tweet_url else None)
                            
                            if like_success:
                                self.file_handler.save_processed_action_key(action_key, timestamp=datetime.now().isoformat())
                                self.processed_action_keys.add(action_key)
                                likes_done_this_run += 1
                                await asyncio.sleep(random.uniform(current_action_config.min_delay_between_actions_seconds / 2, current_action_config.max_delay_between_actions_seconds / 2)) # Shorter delay for likes
                            else:
                                logger.warning(f"[{account.account_id}] Failed to like tweet {tweet_to_like.tweet_id}.")
                
                elif current_action_config.like_tweets_from_feed:
                    logger.warning(f"[{account.account_id}] Liking tweets from feed is enabled but not yet implemented.")
                else:
                    logger.info(f"[{account.account_id}] Liking tweets enabled, but no keywords specified and feed liking is off.")
            
            logger.info(f"[{account.account_id}] Finished processing tasks for this account.")

        except Exception as e:
            logger.error(f"[{account.account_id or 'UnknownAccount'}] Unhandled error during account processing: {e}", exc_info=True)
        finally:
            if browser_manager:
                browser_manager.close_driver()
            logger.info(f"--- Finished processing for account: {account.account_id} ---")
            await asyncio.sleep(self.global_settings.get('delay_between_accounts_seconds', 10))

    async def run(self):
        logger.info("Twitter Orchestrator starting...")
        if not self.accounts_data:
            logger.warning("No accounts found in configuration. Orchestrator will exit.")
            return

        tasks = []
        for account_dict in self.accounts_data:
            tasks.append(self._process_account(account_dict))
        
        logger.info(f"Starting concurrent processing for {len(tasks)} accounts.")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            account_id = self.accounts_data[i].get('account_id', f"AccountIndex_{i}")
            if isinstance(result, Exception):
                logger.error(f"Error processing account {account_id}: {result}", exc_info=result)
            else:
                logger.info(f"Successfully completed processing for account {account_id}.")

        logger.info("Twitter Orchestrator finished processing all accounts.")


if __name__ == "__main__":
    orchestrator = TwitterOrchestrator()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Orchestrator run interrupted by user.")
    except Exception as e:
        logger.critical(f"Orchestrator failed with critical error: {e}", exc_info=True)
    finally:
        logger.info("Orchestrator shutdown complete.")
