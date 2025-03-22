import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objs as go


def analyze_tweet_engagement(data):
    # Check if required columns are present
    required_columns = {'username', 'post_id', 'like_count', 'comment_count', 'view_count'}
    if required_columns.issubset(data.columns):
        # Fill NaN values and convert to float
        data[['like_count', 'comment_count', 'view_count']] = data[['like_count', 'comment_count', 'view_count']].fillna(0).astype(float)

        data['total_engagement'] = data['like_count'] + data['comment_count'] + data['view_count']

        top_users = data.groupby('username')['like_count'].sum().nlargest(5).index.tolist()

        data['pair_id'] = data['username'] + '_' + data['post_id'].astype(str)

        # Create a list of all unique user-post pairs
        all_pairs = []
        for _, row in data.iterrows():
            pair_text = f"{row['username']} (Post: {row['post_id']}) - Likes: {int(row['like_count'])}, Comments: {int(row['comment_count'])}, Views: {int(row['view_count'])}"
            all_pairs.append(pair_text)

        # Get default selections - best post for each top user
        default_posts = []
        for user in top_users:
            # Get the highest engagement post for this user
            best_post = data[data['username'] == user].sort_values('like_count', ascending=False).iloc[0]
            pair_text = f"{best_post['username']} (Post: {best_post['post_id']}) - Likes: {int(best_post['like_count'])}, Comments: {int(best_post['comment_count'])}, Views: {int(best_post['view_count'])}"
            default_posts.append(pair_text)

        # User selects username-post pairs using multiselect
        selected_pairs = st.multiselect(
            "Select Users and Posts to Visualize Engagement Metrics",
            options=all_pairs,
            default=default_posts,  # Default to top 5 users' best posts
            help="Select one or more user-post pairs to visualize their engagement metrics."
        )

        if selected_pairs:
            # Extract username and post_id from selection
            selected_users_posts = []

            for pair in selected_pairs:
                # Split the string to extract username and post_id
                username = pair.split(" (Post: ")[0]
                post_id = pair.split(" (Post: ")[1].split(")")[0]
                selected_users_posts.append((username, post_id))

            # Filter the dataframe for selected user-post pairs
            filtered_data = data[
                data.apply(lambda row: any((row['username'] == user and str(row['post_id']) == post)
                                            for user, post in selected_users_posts), axis=1)
            ]

            # Create a unique identifier for radar chart
            filtered_data['identifier'] = filtered_data['username'] + " (Post: " + filtered_data['post_id'].astype(str) + ")"

            # Check if we actually have data for the selected pairs
            if len(filtered_data) > 0:
                # Extract only the metrics for normalization
                metrics_df = filtered_data[['identifier', 'like_count', 'comment_count', 'view_count']].set_index('identifier')

                # Calculate maximum values for each metric for normalization
                max_values = metrics_df.max()

                # Avoid division by zero
                for col in max_values.index:
                    if max_values[col] == 0:
                        max_values[col] = 1

                # Normalize the metrics
                normalized_metrics = metrics_df / max_values

                # Create spider chart
                st.markdown("### Engagement Metrics for Selected Posts")
                fig_engagement_spider = go.Figure()

                for idx in normalized_metrics.index:
                    fig_engagement_spider.add_trace(go.Scatterpolar(
                        r=normalized_metrics.loc[idx].values,
                        theta=['Likes', 'Comments', 'Views'],
                        fill='toself',
                        name=idx
                    ))

                fig_engagement_spider.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]  # Normalized range
                        ),
                        angularaxis=dict(tickfont=dict(size=10))
                    ),
                    title="Post Engagement",
                    template='plotly_dark',
                    showlegend=False  # Hide legend
                )
                st.plotly_chart(fig_engagement_spider)
            else:
                st.write("No data found for the selected user-post pairs.")
        else:
            st.write("No user-post pairs selected. Please select at least one pair.")
    else:
        st.warning("Required columns for engagement analysis are missing in the dataframe.")

def create_donut_charts(data):
    if {'follower_count', 'following_count'}.issubset(data.columns):
        # Define bins and labels with correct lengths
        bins = [0, 1, 2, 6, 11, 101, 201, 501, 2001, float('inf')]
        labels = ['0', '1', '2-5', '6-10', '11-100', '101-200', '201-500', '501-2000', '2000+']

        # Create follower count donut chart
        data['follower_bin'] = pd.cut(data['follower_count'], bins=bins, labels=labels, right=False)
        follower_counts = data['follower_bin'].value_counts().sort_index()
        fig_follower = px.pie(follower_counts, values=follower_counts.values, names=follower_counts.index,
                              title='Follower Count Distribution', hole=0.5,
                              template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_follower)

        # Create following count donut chart
        data['following_bin'] = pd.cut(data['following_count'], bins=bins, labels=labels, right=False)
        following_counts = data['following_bin'].value_counts().sort_index()
        fig_following = px.pie(following_counts, values=following_counts.values, names=following_counts.index,
                               title='Following Count Distribution', hole=0.5,
                               template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_following)
    else:
        st.warning("Required columns for donut charts are missing in the dataframe.")

# Function to create a post creation timeline as an area chart
def create_timeline(data):
    if 'timestamp' in data.columns:
        # Convert Unix epoch time to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        timeline = data.groupby(data['timestamp'].dt.date).size().reset_index(name='count')

        fig = px.area(timeline, x='timestamp', y='count', title='Post Creation Timeline',
                      labels={'timestamp': 'Date', 'count': 'Number of Posts'},
                      template='plotly_dark', color_discrete_sequence=['#636EFA'])  # Modern color
        fig.update_traces(line_color='#636EFA', fillcolor='rgba(99, 110, 250, 0.2)')
        fig.update_layout(yaxis_title='Number of Posts')
        st.plotly_chart(fig)
    else:
        st.warning("The 'timestamp' column is missing in the dataframe.")

# Function to create a word cloud
def create_wordcloud(text, title):
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig = px.imshow(wordcloud, template='plotly_dark', title=title)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig)
    else:
        st.warning("No text data available to create a word cloud.")

def analyze_post_frequency(data):
    if 'username' in data.columns:
        post_frequency = data['username'].value_counts().head(20)
        fig = px.bar(post_frequency, title='Top 20 Users by Post Frequency',
                     labels={'index': 'Username', 'value': 'Number of Posts'},
                     template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Dark24)  # Modern color
        fig.update_layout(showlegend=False)  # Hide legend
        st.plotly_chart(fig)
    else:
        st.warning("The 'username' column is missing in the dataframe.")

# Function to create a bar chart
def create_bar_chart(data, title, xlabel, ylabel):
    if not data.empty:
        fig = px.bar(data, x=data.index, y=data.values, title=title,
                     labels={xlabel: xlabel, 'value': ylabel},
                     template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)  # Different color
        fig.update_layout(yaxis_title=ylabel, showlegend=False)  # Hide legend
        st.plotly_chart(fig)
    else:
        st.warning("No data available to create a bar chart.")

def create_heatmap(data):
    if 'timestamp' in data.columns:
        # Convert Unix epoch time to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        data['day_of_week'] = data['timestamp'].dt.day_name()
        data['hour'] = data['timestamp'].dt.hour

        # Create a pivot table for the heatmap and fill NaN values with 0
        heatmap_data = data.pivot_table(index='day_of_week', columns='hour', aggfunc='size', fill_value=0)

        # Ensure all days of the week and hours are present
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        hours = list(range(24))
        heatmap_data = heatmap_data.reindex(index=days_order, columns=hours, fill_value=0)

        fig = px.imshow(heatmap_data, aspect="auto", title='Heatmap of Posts by Day of Week and Hour',
                        labels=dict(x="Hour of Day", y="Day of Week", color="Number of Posts"),
                        template='plotly_dark', color_continuous_scale="Viridis")
        st.plotly_chart(fig)
    else:
        st.warning("The 'timestamp' column is missing in the dataframe.")

# Streamlit app
def main():
    st.title('Social Media Post Analysis')

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Post creation timeline
        st.subheader('Post Creation Timeline')
        create_timeline(data)

        # Word cloud of captions
        st.subheader('Word Cloud of Captions')
        if 'caption' in data.columns:
            captions = ' '.join(data['caption'].dropna().astype(str))
            create_wordcloud(captions, 'Word Cloud of Captions')
        else:
            st.warning("The 'caption' column is missing in the dataframe.")

        st.subheader('Heatmap of Posts by Day of Week and Hour')
        create_heatmap(data)

        # Word cloud of hashtags
        st.subheader('Word Cloud of Hashtags')
        if 'hashtags' in data.columns:
            hashtags = ' '.join(data['hashtags'].dropna().astype(str))
            create_wordcloud(hashtags, 'Word Cloud of Hashtags')
        else:
            st.warning("The 'hashtags' column is missing in the dataframe.")

        analyze_tweet_engagement(data)

        # Top 20 users by number of posts
        st.subheader('Top 20 Users by Number of Posts')
        analyze_post_frequency(data)

        # Top 20 mentions
        st.subheader('Top 20 Mentions')
        if 'mentions' in data.columns:
            mentions = ' '.join(data['mentions'].dropna().astype(str)).split()
            top_mentions = Counter(mentions).most_common(20)
            top_mentions_df = pd.DataFrame(top_mentions, columns=['Mention', 'Count']).set_index('Mention')
            create_bar_chart(top_mentions_df['Count'], 'Top 20 Mentions', 'Mention', 'Count')
        else:
            st.warning("The 'mentions' column is missing in the dataframe.")

        st.subheader('Follower and Following Count Distribution')
        create_donut_charts(data)

if __name__ == '__main__':
    main()
