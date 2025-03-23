import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objs as go
from datetime import datetime
def classify_users_by_post_frequency(data):
    # Check if required columns are present
    required_columns = {'username', 'post_count', 'user_creation_date'}
    if required_columns.issubset(data.columns):
        # Convert user_creation_date from Unix to datetime
        data['user_creation_date'] = pd.to_datetime(data['user_creation_date'], unit='s', errors='coerce')

        # Get the current date
        current_date = datetime.now()

        # Calculate the number of days since user creation
        data['days_since_creation'] = (current_date - data['user_creation_date']).dt.days

        # Calculate daily post rate
        data['daily_post_rate'] = data['post_count'] / data['days_since_creation']

        # Classify users based on daily post rate
        def classify_user(rate):
            if rate > 10:
                return 'Bot'
            elif 5 < rate <= 10:
                return 'Potential Bot'
            else:
                return 'Need More Analysis'

        data['user_classification'] = data['daily_post_rate'].apply(classify_user)

       
        st.write(data[['username', 'post_count', 'user_creation_date', 'days_since_creation', 'daily_post_rate', 'user_classification']])
    else:
        st.warning("Required columns for user classification are missing in the dataframe.")
def create_user_creation_timeline(data):
    # Check if required column is present
    if 'user_creation_date' in data.columns:
        # Convert user_creation_date from Unix to datetime
        data['user_creation_date'] = pd.to_datetime(data['user_creation_date'], unit='s', errors='coerce')

        # Group by user creation date and count unique users
        user_creation_timeline = data.groupby(data['user_creation_date'].dt.date).size().reset_index(name='user_count')

        # Create area chart
        fig = px.area(user_creation_timeline, x='user_creation_date', y='user_count', title='User Creation Timeline',
                       labels={'user_creation_date': 'Date', 'user_count': 'Number of Users Created'},
                       template='plotly_dark')
        fig.update_traces(line_color='#b30743', fillcolor='rgba(171, 22, 74, 0.2)')
        fig.update_layout(yaxis_title='Number of Users Created')
        st.plotly_chart(fig)
    else:
        st.warning("The 'user_creation_date' column is missing in the dataframe.")
def classify_posts_by_hashtags(data):
    # Check if required column is present
    if 'hashtags' in data.columns:
        # Count the number of hashtags in each post
        data['hashtag_count'] = data['hashtags'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)

        # Classify posts based on hashtag count
        def classify_post(count):
            if count > 10:
                return 'Bot'
            elif 5 < count <= 10:
                return 'Potential Bot'
            else:
                return 'Needs More Analysis'

        data['post_classification'] = data['hashtag_count'].apply(classify_post)

        # Display the classification results
        st.subheader('Post Classification Based on Hashtag Count')
        st.write(data[['username', 'hashtags', 'hashtag_count', 'post_classification']])
    else:
        st.warning("The 'hashtags' column is missing in the dataframe.")

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
        wordcloud = WordCloud(width=800, height=400, background_color='white',font_path='JustAnotherHand-Regular.ttf').generate(text)
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
def analyze_following_follower_ratio(dataframe):
    # Check if required columns are present
    if 'follower_count' in dataframe.columns and 'following_count' in dataframe.columns:
        # Convert columns to numeric and handle errors
        dataframe['follower_count'] = pd.to_numeric(dataframe['follower_count'], errors='coerce').fillna(0)
        dataframe['following_count'] = pd.to_numeric(dataframe['following_count'], errors='coerce').fillna(0)

        # Calculate the following-to-follower ratio
        dataframe['following_follower_ratio'] = dataframe.apply(
            lambda row: row['following_count'] / (row['follower_count'] if row['follower_count'] != 0 else 1),
            axis=1
        )

        # Function to classify users based on the ratio and followers
        def classify_user(row):
            if row['follower_count'] < 10:
                # Adjusted conditions for followers < 10
                if row['following_follower_ratio'] > 100:
                    return 'Bot'
                elif 50 < row['following_follower_ratio'] <= 100:
                    return 'Potential Bot'
                else:
                    return 'Need More Analysis'
            else:
                # Normal conditions for followers >= 10
                if row['following_follower_ratio'] > 25:
                    return 'Bot'
                elif 15 <= row['following_follower_ratio'] <= 25:
                    return 'Potential Bot'
                else:
                    return 'Need More Analysis'

       
        dataframe['following_follower_label'] = dataframe.apply(classify_user, axis=1)

        
        st.markdown("### Following-to-Follower Ratio Analysis")
        st.write(dataframe[['username', 'follower_count', 'following_count', 'following_follower_ratio', 'following_follower_label']])

    
        label_counts = dataframe['following_follower_label'].value_counts().reset_index()
        label_counts.columns = ['Classification', 'Count']

        
        ratio_csv = dataframe[['username', 'follower_count', 'following_count', 'following_follower_ratio', 'following_follower_label']].to_csv(index=False).encode('utf-8')
        
        
    else:
        st.write("Columns 'follower_count' or 'following_count' not found in the data.")
def comprehensive_bot_analysis(dataframe):
    
    def final_classification(row):
        labels = [row['following_follower_label'], row['post_classification'], row['user_classification']]
        if 'Bot' in labels:
            return 'Bot'
        elif 'Potential Bot' in labels:
            return 'Potential Bot'
        else:
            return 'Need More Analysis'

    dataframe['final_bot_classification'] = dataframe.apply(final_classification, axis=1)

    # Display the final classification results
    st.markdown("### Comprehensive Bot Analysis Results")
    st.write(dataframe[['username','follower_count','following_count','post_count','daily_post_rate','final_bot_classification']])

    # Generate bar chart of final classifications
    final_label_counts = dataframe['final_bot_classification'].value_counts().reset_index()
    final_label_counts.columns = ['Classification', 'Count']

    fig = px.bar(final_label_counts, x='Classification', y='Count', title='User Classification',
                 labels={'Classification': 'Classification', 'Count': 'Number of Users'},
                 template='plotly_dark', color='Classification',  # This assigns a different color to each category
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig)

    # Allow download of the data with a unique key
    final_csv = dataframe[['username','final_bot_classification']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Comprehensive Bot Analysis Data as CSV",
        data=final_csv,
        file_name="comprehensive_bot_analysis_data.csv",
        mime="text/csv",
        key="download_comprehensive_data"
    )
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

        st.subheader('User Creation Timeline')
        create_user_creation_timeline(data)

        st.subheader('User Classification Based on Post Frequency')
        classify_users_by_post_frequency(data)
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
        
        classify_posts_by_hashtags(data)

        analyze_tweet_engagement(data)


        # Top 20 users by number of posts
        st.subheader('Top 20 Users by Number of Posts')
        analyze_post_frequency(data)

        analyze_following_follower_ratio(data)

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

        comprehensive_bot_analysis(data)

if __name__ == '__main__':
    main()
