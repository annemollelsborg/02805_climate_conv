import { Client } from "twitter-api-sdk";
import "dotenv/config";
import fs from "fs";
import path from "path";

// --- helper: timestamped filename
function stamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

// --- helper: sleep (ms)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const client = new Client(process.env.BEARER_TOKEN);
  const allTweets = [];
  let nextToken = null;
  let page = 0;

  const baseParams = {
    query: '(#climatechange OR "climate change" OR #climate OR "global warming" OR climate OR carbon OR emissions) (is:reply OR has:mentions OR is:quote) lang:en -is:retweet',
    start_time: "2025-11-16T00:00:00.000Z",
    end_time: "2025-11-17T00:00:00.000Z",
    max_results: 100,
    sort_order: "relevancy",
    "tweet.fields": [
      "attachments","author_id","conversation_id","created_at","entities",
      "id","in_reply_to_user_id","lang","public_metrics","referenced_tweets","text"
    ],
    expansions: [
      "author_id","entities.mentions.username","in_reply_to_user_id",
      "referenced_tweets.id","referenced_tweets.id.author_id"
    ],
    "user.fields": [
      "created_at","description","id","name","public_metrics","username","verified"
    ],
    "place.fields": ["country","country_code"]
  };

  try {
    // aim for roughly 10 pages (~1000 tweets)
    while (page < 10) {
      const params = nextToken ? { ...baseParams, next_token: nextToken } : baseParams;
      console.log(`🔹 Fetching page ${page + 1}...`);

      const response = await client.tweets.tweetsRecentSearch(params);
      const tweets = response.data ?? [];
      allTweets.push(...tweets);

      console.log(`   → Got ${tweets.length} tweets (total ${allTweets.length})`);

      nextToken = response.meta?.next_token;
      if (!nextToken) {
        console.log("✅ No more pages available (less than 1000 tweets total).");
        break;
      }

      page++;

      // free tier: wait 16 minutes between calls (15 + 1 buffer)
      if (page < 10) {
        console.log("⏳ Waiting 16 minutes before next request to respect rate limit...");
        await sleep(16 * 60 * 1000);
      }
    }

    // Save all tweets at once
    const fname = `tweets_climatechange_${stamp()}.json`;
    const outPath = path.join(process.cwd(), fname);
    fs.writeFileSync(outPath, JSON.stringify(allTweets, null, 2));
    console.log(`💾 Saved ${allTweets.length} tweets to ${fname}`);

  } catch (e) {
    if (e.status === 429) {
      console.error("⚠️ Rate limit hit (HTTP 429). Wait 15+ minutes before retrying.");
    } else if (e.status === 400) {
      console.error("⚠️ Bad request (HTTP 400). Check your dates and query syntax.");
    } else {
      console.error("❌ Error:", e);
    }
  }
}

main();
